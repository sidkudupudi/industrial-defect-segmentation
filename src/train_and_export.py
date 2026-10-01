import torch
import os
from ultralytics import YOLO

def build_pipeline():
    print("Starting Training...")
    model = YOLO('yolo11l-seg.pt')
    
    # Train the model
    model.train(
        data='./visa_yolo/visa.yaml',
        epochs=100,
        imgsz=640,
        device=0,
        batch=16,
        project='industrial_pipeline',
        name='defect_seg'
    )
    
    # Export immediately after training
    print("Training complete. Exporting to TensorRT...")
    best_weights = 'industrial_pipeline/defect_seg/weights/best.pt'
    
    if os.path.exists(best_weights):
        best_model = YOLO(best_weights)
        best_model.export(
            format='engine',
            device=0,
            half=True,
            dynamic=False,
            workspace=8 
        )
        print("Export successful! Your .engine file is ready.")
    else:
        print(f"Error: Could not find weights at {best_weights}")

if __name__ == '__main__':
    build_pipeline()