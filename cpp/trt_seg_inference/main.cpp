#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace nvinfer1;

// --- UPDATED 1024px DIMENSIONS ---
const int INPUT_W = 1024;
const int INPUT_H = 1024;
const int BATCH_SIZE = 1;
const int NUM_CLASSES = 15;
const int MASK_PROTOS = 32;
const int NUM_PROPOSALS = 21504; 

const int OUTPUT0_SIZE = BATCH_SIZE * (4 + NUM_CLASSES + MASK_PROTOS) * NUM_PROPOSALS; 
const int OUTPUT1_SIZE = BATCH_SIZE * MASK_PROTOS * 256 * 256; 

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} logger;

int main() {
    // 1. LOAD THE NEW 1024px ENGINE
    std::ifstream file("best_1024.engine", std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Could not find best_1024.engine! Ensure it compiled successfully." << std::endl;
        return -1;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine) {
        // Engines are tied to the TensorRT version that built them; after a TensorRT upgrade, rebuild from the ONNX:
        //   trtexec --onnx=best.onnx --saveEngine=best_1024.engine
        std::cerr << "Error: could not deserialize best_1024.engine (built with a different TensorRT version?)." << std::endl;
        return -1;
    }
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error: could not create a TensorRT execution context." << std::endl;
        return -1;
    }

    // 2. ALLOCATE ASYNC RESOURCES
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *hostInputBuffer, *hostOutput0Buffer, *hostOutput1Buffer;
    cudaHostAlloc((void**)&hostInputBuffer, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostOutput0Buffer, OUTPUT0_SIZE * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostOutput1Buffer, OUTPUT1_SIZE * sizeof(float), cudaHostAllocDefault);

    void* buffers[3];
    cudaMalloc(&buffers[0], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)); 
    cudaMalloc(&buffers[1], OUTPUT0_SIZE * sizeof(float));                       
    cudaMalloc(&buffers[2], OUTPUT1_SIZE * sizeof(float));                       

    // 3. OPENCV PREPROCESSING
    cv::Mat img = cv::imread("test_defect.png");
    if (img.empty()) {
        std::cerr << "Error: Could not load test_defect.png" << std::endl;
        return -1;
    }

    cv::Mat resized, rgb, normalized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(normalized, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; ++i) {
        chw[i] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, hostInputBuffer + i * INPUT_H * INPUT_W);
    }
    cv::split(normalized, chw);

    // 4. ASYNCHRONOUS INFERENCE EXECUTION
    std::cout << "Executing Extra-Large 1024px TensorRT Engine..." << std::endl;
    cudaMemcpyAsync(buffers[0], hostInputBuffer, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    context->setTensorAddress("images", buffers[0]);
    context->setTensorAddress("output0", buffers[1]);
    context->setTensorAddress("output1", buffers[2]);
    context->enqueueV3(stream);

    cudaMemcpyAsync(hostOutput0Buffer, buffers[1], OUTPUT0_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(hostOutput1Buffer, buffers[2], OUTPUT1_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);

    // 5. POST-PROCESSING & NMS
    cv::Mat output0_mat(4 + NUM_CLASSES + MASK_PROTOS, NUM_PROPOSALS, CV_32F, hostOutput0Buffer);
    cv::Mat proposals = output0_mat.t(); 

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<std::vector<float>> maskWeights;

    float scaleX = (float)img.cols / INPUT_W;
    float scaleY = (float)img.rows / INPUT_H;

    for (int i = 0; i < NUM_PROPOSALS; i++) {
        float* row = proposals.ptr<float>(i);
        cv::Mat scores(1, NUM_CLASSES, CV_32F, row + 4);
        cv::Point classIdPoint;
        double maxScore;
        cv::minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

        if (maxScore > 0.5) { 
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];
            
            int left = int((cx - 0.5 * w) * scaleX);
            int top = int((cy - 0.5 * h) * scaleY);
            int width = int(w * scaleX);
            int height = int(h * scaleY);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back((float)maxScore);
            classIds.push_back(classIdPoint.x);

            std::vector<float> weight(row + 4 + NUM_CLASSES, row + 4 + NUM_CLASSES + MASK_PROTOS);
            maskWeights.push_back(weight);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

    // 6. DRAWING MASKS AND BOXES (UPDATED TO 256x256)
    cv::Mat protos(MASK_PROTOS, 256 * 256, CV_32F, hostOutput1Buffer);
    
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        box &= cv::Rect(0, 0, img.cols, img.rows);
        
        cv::Mat maskWeight(1, MASK_PROTOS, CV_32F, maskWeights[idx].data());
        cv::Mat mask = maskWeight * protos; 
        mask = mask.reshape(1, 256); // Reshape to the new 256 resolution

        cv::exp(-mask, mask);
        mask = 1.0f / (1.0f + mask);

        cv::resize(mask, mask, img.size());
        mask = mask(box) > 0.5f; 

        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);

        cv::Mat coloredMask = cv::Mat::zeros(img.size(), img.type());
        coloredMask(box).setTo(cv::Scalar(0, 0, 255), mask);
        cv::addWeighted(img, 1.0, coloredMask, 0.5, 0.0, img);
        
        std::cout << "High-Res Defect Detected! Class ID: " << classIds[idx] << " | Confidence: " << confidences[idx] << std::endl;
    }

    cv::imwrite("final_output_1024.jpg", img);
    std::cout << "Inference Complete. Saved as 'final_output_1024.jpg'." << std::endl;

    // 7. CLEANUP
    cudaFreeHost(hostInputBuffer);
    cudaFreeHost(hostOutput0Buffer);
    cudaFreeHost(hostOutput1Buffer);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
