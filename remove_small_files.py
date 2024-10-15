import os
from PIL import Image

# Define image directory
image_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\train_images'

# Iterate through images and remove corrupt or small images
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        try:
            with Image.open(image_path) as img:
                if img.size[0] < 10 or img.size[1] < 10:
                    print(f"Removing small or corrupt image: {filename} (size: {img.size})")
                    os.remove(image_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
