import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize

# Define the directory containing the NetCDF files and the output directory for images
input_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\Lightning_Project\pyltg\pyltg\examples\train_files'
output_dir = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\train_images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over all .nc files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.nc'):
        # Construct full file path
        file_path = os.path.join(input_dir, filename)
        
        # Load the NetCDF file
        dataset = xr.open_dataset(file_path)

        # Get relevant data: latitude, longitude, radiance
        lat = dataset['lightning_event_lat'].values
        lon = dataset['lightning_event_lon'].values
        radiance = dataset['lightning_event_radiance'].values

        # Define grid resolution
        grid_res = 0.1  # Set resolution of the grid (degree)
        lat_bins = np.arange(min(lat), max(lat), grid_res)
        lon_bins = np.arange(min(lon), max(lon), grid_res)

        # Create a 2D histogram as an image, weighted by radiance
        heatmap, _, _ = np.histogram2d(lat, lon, bins=[lat_bins, lon_bins], weights=radiance)

        # Normalize the heatmap manually
        heatmap_normalized = Normalize(vmin=heatmap.min(), vmax=heatmap.max())(heatmap)

        # Create output image path
        output_image = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.png')

        # Save the heatmap as an image without using the 'norm' argument
        plt.imsave(output_image, heatmap_normalized, cmap='plasma')

        print(f"Created image: {output_image}")

print("All images have been created.")

