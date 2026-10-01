import os
import random

train_images_dir = './visa_yolo/images/train'
train_labels_dir = './visa_yolo/labels/train'

# Find all images in the training folder
all_images = os.listdir(train_images_dir)

# Separate normal (background) images from anomaly images
normal_images = [img for img in all_images if '_normal_' in img]
anomaly_images = [img for img in all_images if '_anomaly_' in img]

print(f"Before balancing: {len(anomaly_images)} Defects | {len(normal_images)} Backgrounds")

# Calculate how many normal images we want to keep (10% of anomaly count)
target_normal_count = int(len(anomaly_images) * 0.10)

# Randomly shuffle and select the ones to delete
random.shuffle(normal_images)
images_to_delete = normal_images[target_normal_count:]

# Delete the excess background images
for img in images_to_delete:
    os.remove(os.path.join(train_images_dir, img))

print(f"Deleted {len(images_to_delete)} excess background images.")
print(f"After balancing: {len(anomaly_images)} Defects | {target_normal_count} Backgrounds")