from ultralytics import YOLO
import os

best_weights = 'runs/segment/industrial_pipeline/defect_seg-2/weights/best.pt'

if os.path.exists(best_weights):
    print("Found weights! Compiling TensorRT Engine ...")
    model = YOLO(best_weights)
    model.export(
        format='engine',
        device=0,
        half=True,
        dynamic=False,
        workspace=8 
    )
    print("Export successful!")
else:
    print("Error: Could not find weights. Check the path!")
