
import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Directory where your NetCDF TRMM LIS files are stored
directory_path = "C:/Users/jbull/OneDrive - Fayetteville State University/CSC490 SENIOR PROJECT/Lightning_Project/pyltg/pyltg/examples/test_files/"

# List all NetCDF files in the directory
file_names = [file for file in os.listdir(directory_path) if file.endswith(".nc")]

# Initialize lists to store combined data
lat_data_list = []
lon_data_list = []
time_data_list = []

# Loop over each file, open the dataset, and extract data
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
            # Clean and append the data (remove NaN and invalid values)
            lat_data_clean = lat_data[~np.isnan(lat_data)]
            lon_data_clean = lon_data[~np.isnan(lon_data)]
            time_data_clean = time_data[~np.isnan(time_data)]
            
            valid_lat = (lat_data_clean >= -90) & (lat_data_clean <= 90)
            valid_lon = (lon_data_clean >= -180) & (lon_data_clean <= 180)
            
            lat_data_filtered = lat_data_clean[valid_lat & valid_lon]
            lon_data_filtered = lon_data_clean[valid_lat & valid_lon]
            time_data_filtered = time_data_clean[valid_lat & valid_lon]
            
            # Append filtered data to the lists
            lat_data_list.append(lat_data_filtered)
            lon_data_list.append(lon_data_filtered)
            time_data_list.append(time_data_filtered)
            
        # Close the dataset after processing
        dataset.close()
        
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine data from all files into single arrays
lat_data_all = np.concatenate(lat_data_list)
lon_data_all = np.concatenate(lon_data_list)
time_data_all = np.concatenate(time_data_list)

# Plot the combined data with Earth as the background
plt.figure(figsize=(12, 8))

# Set up the projection for the plot (using PlateCarree for regular lat/lon)
ax = plt.axes(projection=ccrs.PlateCarree())

# Add coastlines and optional features (like borders, land, and water features)
ax.coastlines(resolution='110m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Plot the lightning events on the map
ax.scatter(lon_data_all, lat_data_all, s=1, c='red', alpha=0.5, transform=ccrs.PlateCarree())

# Set title and labels
plt.title('Combined Spatial Distribution of Lightning Events (All TRMM Datasets)', fontsize=15)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Display the plot
plt.show()
