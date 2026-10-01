cmake_minimum_required(VERSION 3.10)
project(DefectInference)

set(CMAKE_CXX_STANDARD 17)

# Find necessary packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Standard Ubuntu paths for apt-get installed TensorRT
include_directories(/usr/include/x86_64-linux-gnu /usr/include)
link_directories(/usr/lib/x86_64-linux-gnu /usr/lib)

add_executable(inference main.cpp)

# Link libraries
target_include_directories(inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference nvinfer cudart ${OpenCV_LIBS})
