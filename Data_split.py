import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
image_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\train_images'
label_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\train_labels'
train_image_dir = r'dataset/images/train'
val_image_dir = r'dataset/images/val'
train_label_dir = r'dataset/labels/train'
val_label_dir = r'dataset/labels/val'

# Ensure directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get all image filenames
images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Split into train and validation sets (80% train, 20% validation)
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# Move files to their respective directories
for image in train_images:
    shutil.move(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
    label = image.replace('.png', '.txt')
    shutil.move(os.path.join(label_dir, label), os.path.join(train_label_dir, label))

for image in val_images:
    shutil.move(os.path.join(image_dir, image), os.path.join(val_image_dir, image))
    label = image.replace('.png', '.txt')
    shutil.move(os.path.join(label_dir, label), os.path.join(val_label_dir, label))

print("Dataset split and moved successfully.")
