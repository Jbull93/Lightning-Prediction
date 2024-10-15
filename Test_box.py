import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize

# Define the directories for input and output
input_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\Lightning_Project\pyltg\pyltg\examples\train_files'
image_output_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\train_images'
label_output_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\train_labels'

# Ensure the output directories exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

# Iterate over all .nc files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.nc'):
        # Construct full file path
        file_path = os.path.join(input_dir, filename)
        
        # Load the NetCDF file
        dataset = xr.open_dataset(file_path)

        # Extract latitude, longitude, and radiance for lightning events
        lat = dataset['lightning_flash_lat'].values
        lon = dataset['lightning_flash_lon'].values
        radiance = dataset['lightning_flash_radiance'].values

        # Define image size (e.g., 640x640 pixels)
        image_width = 640
        image_height = 640

        # Define geographic bounds for your data
        min_lat, max_lat = lat.min(), lat.max()
        min_lon, max_lon = lon.min(), lon.max()

        # Convert lat/lon to pixel coordinates based on image dimensions
        x_pixels = ((lon - min_lon) / (max_lon - min_lon)) * image_width
        y_pixels = ((lat - min_lat) / (max_lat - min_lat)) * image_height

        # Normalize radiance values to adjust bounding box size
        radiance_min = radiance.min()
        radiance_max = radiance.max()
        box_sizes = ((radiance - radiance_min) / (radiance_max - radiance_min)) * 20  # Scale box size with radiance

        # Create a 2D histogram as an image, weighted by radiance
        grid_res = 0.1  # Set grid resolution (degree)
        lat_bins = np.arange(min(lat), max(lat), grid_res)
        lon_bins = np.arange(min(lon), max(lon), grid_res)
        heatmap, _, _ = np.histogram2d(lat, lon, bins=[lat_bins, lon_bins], weights=radiance)
        heatmap_normalized = Normalize(vmin=heatmap.min(), vmax=heatmap.max())(heatmap)

        # Create output image path and save heatmap
        output_image = os.path.join(image_output_dir, f'{os.path.splitext(filename)[0]}.png')
        plt.imsave(output_image, heatmap_normalized, cmap='plasma')
        print(f"Created image: {output_image}")

        # Create bounding boxes in YOLO format
        bounding_boxes = []
        for x, y, box_size in zip(x_pixels, y_pixels, box_sizes):
            x_center = x / image_width  # Normalize between 0 and 1
            y_center = y / image_height  # Normalize between 0 and 1
            width = box_size / image_width  # Normalize box width
            height = box_size / image_height  # Normalize box height
            
            # Append bounding box in YOLO format (class 0 for lightning strikes)
            bounding_boxes.append(f"0 {x_center} {y_center} {width} {height}")

        # Save bounding boxes to YOLO format
        output_file = os.path.join(label_output_dir, f'{os.path.splitext(filename)[0]}.txt')
        with open(output_file, 'w') as f:
            for box in bounding_boxes:
                f.write(box + '\n')
        print(f"Bounding boxes saved to {output_file}")
