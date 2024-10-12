import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import csv

# Directory where your NetCDF TRMM LIS files are stored
directory_path = "C:/Users/jbull/OneDrive - Fayetteville State University/CSC490 SENIOR PROJECT/Lightning_Project/pyltg/pyltg/examples/test_files/"

# Output file for coordinates with timestamps
csv_output_file = "lightning_coordinates_with_timestamps.csv"

# List all NetCDF files in the directory
file_names = [file for file in os.listdir(directory_path) if file.endswith(".nc")]

# Initialize lists to store the data (latitude, longitude, and time)
lat_data_list = []
lon_data_list = []
time_data_list = []

# Function to convert TAI93 time to seconds (since 1993)
def convert_tai93_to_seconds(tai93_time):
    base_time = 0  # TAI93 epoch is January 1, 1993
    return tai93_time + base_time

# Loop over each file, open the dataset, and extract latitude, longitude, and time
for file in file_names:
    file_path = os.path.join(directory_path, file)
    
    # Open the NetCDF dataset
    try:
        dataset = nc.Dataset(file_path, mode='r')
        
        # Extract latitude, longitude, and time variables
        lat_data = dataset.variables.get('lightning_event_lat', None)
        lon_data = dataset.variables.get('lightning_event_lon', None)
        time_data = dataset.variables.get('lightning_event_TAI93_time', None)
        
        if lat_data is not None and lon_data is not None and time_data is not None:
            # Clean and append the data (remove NaN values)
            lat_data_clean = lat_data[~np.isnan(lat_data)]
            lon_data_clean = lon_data[~np.isnan(lon_data)]
            time_data_clean = time_data[~np.isnan(time_data)]
            
            # Convert time to seconds
            time_data_seconds = convert_tai93_to_seconds(time_data_clean)
            
            # Append the filtered data to the lists
            lat_data_list.append(lat_data_clean)
            lon_data_list.append(lon_data_clean)
            time_data_list.append(time_data_seconds)
        
        # Close the dataset after processing
        dataset.close()
        
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine data from all files into single arrays
lat_data_all = np.concatenate(lat_data_list)
lon_data_all = np.concatenate(lon_data_list)
time_data_all = np.concatenate(time_data_list)

# Sort the data by time to visualize the temporal progression
sorted_indices = np.argsort(time_data_all)
lat_data_sorted = lat_data_all[sorted_indices]
lon_data_sorted = lon_data_all[sorted_indices]
time_data_sorted = time_data_all[sorted_indices]

# Create a GeoPandas DataFrame from the sorted data
geometry = [Point(lon, lat) for lon, lat in zip(lon_data_sorted, lat_data_sorted)]
gdf = gpd.GeoDataFrame(geometry=geometry)

# Prepare to collect coordinates and timestamps for saving in CSV
coordinates_with_timestamps = []

# Create a figure with a Cartopy map projection
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Set the extent (view) of the map based on the data
ax.set_extent([np.min(lon_data_sorted), np.max(lon_data_sorted), np.min(lat_data_sorted), np.max(lat_data_sorted)], crs=ccrs.PlateCarree())

# Add Earth features: coastlines, land, and borders
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Use Contextily to add a satellite basemap
ax = gdf.plot(ax=ax, color='red', markersize=5, transform=ccrs.PlateCarree())
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=ccrs.PlateCarree())

# Function to initialize the animation
scatter = ax.scatter([], [], s=10, c='yellow', transform=ccrs.PlateCarree())

def init():
    scatter.set_offsets(np.empty((0, 2)))
    return scatter,

# Function to update the scatter plot in each frame
def update(frame):
    # Prepare data for the current frame (lon, lat pairs)
    data = np.column_stack((lon_data_sorted[:frame], lat_data_sorted[:frame]))
    scatter.set_offsets(data)
    
    # Save coordinates and corresponding time for the current frame
    if frame < len(time_data_sorted):
        coordinates_with_timestamps.append([lon_data_sorted[frame], lat_data_sorted[frame], time_data_sorted[frame]])
    
    return scatter,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time_data_sorted), init_func=init, blit=True, interval=50)

# Show the animation
plt.title('Lightning Events on Satellite Earth Map')
plt.show()

# Save the coordinates and timestamps into a CSV file
with open(csv_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Longitude', 'Latitude', 'Timestamp'])
    writer.writerows(coordinates_with_timestamps)

print(f"Coordinates with timestamps saved to {csv_output_file}")

# Optionally, save the animation as an mp4 file
ani.save('lightning_animation_with_satellite.mp4', writer='ffmpeg', fps=30)
