from ultralytics import YOLO

print("Initializing Ultimate Production Training Run...")

# 1. Load the Extra Large Model
model = YOLO('yolo11x-seg.pt')

# 2. Train using your mathematically evolved hyperparameters
model.train(
    data='./mvtec_yolo/mvtec.yaml',
    cfg='./runs/segment/tune-2/best_hyperparameters.yaml', # Inject the golden settings
    epochs=150,
    imgsz=1024,
    batch=8,      
    device=0,
    workers=8,
    project='industrial_pipeline',
    name='mvtec_ultimate_seg'
)

print("Ultimate Model Training Complete!")