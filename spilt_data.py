import os
import random
import shutil
from pathlib import Path

# original data directory
data_dir = Path('data')
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'

for split_dir in [train_dir, test_dir]:
    split_dir.mkdir(parents=True, exist_ok=True)

# every class directory under data_dir
for class_dir in data_dir.iterdir():
    if class_dir.is_dir() and class_dir.name not in ['train', 'test']:
        images = list(class_dir.glob('*.*'))  # retrieve all image files in the class directory
        random.shuffle(images)  # randomly shuffle the images

        split_idx = int(0.8 * len(images))  # 80% for training, 20% for testing
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # make sure the train and test directories for the class exist
        (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

        # copy images to the respective directories
        for img in train_images:
            shutil.move(img, train_dir / class_dir.name / img.name)
        for img in test_images:
            shutil.move(img, test_dir / class_dir.name / img.name)
print("data split completed successfully!")
