import os
import cv2
import random
import shutil

def format_mvtec_to_yolo(source_dir, dest_dir, split_ratio=0.8):
    # MVTec categories = folders that contain a test/ split (sorted so class ids are stable across runs;
    # other folders in the working directory must not receive a class id)
    classes = sorted(d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d, 'test')))
    
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)

    yaml_classes = "path: .\ntrain: images/train\nval: images/val\n\nnames:\n"

    for class_id, obj_class in enumerate(classes):
        yaml_classes += f"  {class_id}: {obj_class}\n"
        test_dir = os.path.join(source_dir, obj_class, 'test')
        gt_dir = os.path.join(source_dir, obj_class, 'ground_truth')
        
        if not os.path.exists(test_dir): continue

        defect_types = [d for d in os.listdir(test_dir) if d != 'good']
        
        for defect in defect_types:
            img_folder = os.path.join(test_dir, defect)
            gt_folder = os.path.join(gt_dir, defect)
            images = os.listdir(img_folder)
            random.shuffle(images)
            split_idx = int(len(images) * split_ratio)
            train_imgs = set(images[:split_idx])
            
            for img_name in images:
                subset = 'train' if img_name in train_imgs else 'val'
                new_name = f"{obj_class}_{defect}_{img_name}"
                shutil.copy(os.path.join(img_folder, img_name), os.path.join(dest_dir, 'images', subset, new_name))
                
                mask_path = os.path.join(gt_folder, img_name.split('.')[0] + '_mask.png')
                label_path = os.path.join(dest_dir, 'labels', subset, new_name.replace('.png', '.txt'))
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    h, w = mask.shape
                    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    with open(label_path, 'w') as f:
                        for contour in contours:
                            if len(contour) > 2:
                                line = f"{class_id} " + " ".join([f"{p[0][0]/w:.6f} {p[0][1]/h:.6f}" for p in contour])
                                f.write(line + '\n')

    with open(os.path.join(dest_dir, 'mvtec.yaml'), 'w') as f:
        f.write(yaml_classes)

if __name__ == '__main__':
    # run from the MVTec AD root (the folder that contains bottle/, cable/, ...)
    format_mvtec_to_yolo('./', './mvtec_yolo')