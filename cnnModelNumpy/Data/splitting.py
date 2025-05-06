import os
import shutil
import random
from pathlib import Path

# for splitting malignant vs benign only
mag = "200X"
base_dir = Path(__file__).resolve().parent

source_dir = base_dir / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
train_dir = base_dir / f"dataset_split2_{mag}" / "train"
val_dir = base_dir / f"dataset_split2_{mag}" / "val"
test_dir = base_dir / f"dataset_split2_{mag}" / "test"

categories = ["benign", "malignant"]

for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

def collect_images(root_dir, category, mag="200X"):
    image_paths = []

    category_dir = os.path.join(root_dir, category)

    if not os.path.exists(category_dir):
        print(f"Directory does not exist: {category_dir}")
        return image_paths
    
    for root, dirs, files, in os.walk(category_dir):
        print(f"Checking directory: {root}")
        if mag in root:
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, file))
                    print(f"Found image file: {file}")
    print(f"Found {len(image_paths)} images in {root_dir}.")
    return image_paths

def move_images(image_list, target_folder):
    for img_path in image_list:
        if os.path.exists(img_path):
            filename = os.path.basename(img_path)
            target_path = os.path.join(target_folder, filename)
            
            try:
                shutil.copy(img_path, target_path)
                print(f"Copied: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        else:
            print(f"Image file does not exist: {img_path}")

total_test = 0
total_train = 0
total_val = 0

for category in categories:
    print(f"Processing {category}")

    images = collect_images(source_dir, category)

    random.shuffle(images)

    train_ratio = 0.8
    val_ratio = 0.1

    train_split = int(len(images) * train_ratio)
    val_split = int(len(images) * (train_ratio + val_ratio))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    move_images(train_images, os.path.join(train_dir, category))
    move_images(val_images, os.path.join(val_dir, category))
    move_images(test_images, os.path.join(test_dir, category))

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

print("completed")
print(f"Train: {total_train} images")
print(f"Validation: {total_val} images")
print(f"Test: {total_test} images")