import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Directory where your NetCDF TRMM LIS files are stored
directory_path = "C:/Users/jbull/OneDrive - Fayetteville State University/CSC490 SENIOR PROJECT/Lightning_Project/pyltg/pyltg/examples/test_files/"

# List all NetCDF files in the directory
file_names = [file for file in os.listdir(directory_path) if file.endswith(".nc")]

# Initialize lists to store the data for clustering (latitude, longitude, and time)
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
            
            # Convert time to seconds (optional: use datetime if needed)
            time_data_seconds = convert_tai93_to_seconds(time_data_clean)
            
            # Append filtered data to the lists
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

# Combine latitude, longitude, and time into a single array for clustering
data_for_clustering = np.column_stack((lat_data_all, lon_data_all, time_data_all))

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=10).fit(data_for_clustering)

# Get the labels (clusters) from DBSCAN
labels = dbscan.labels_

# Number of clusters (excluding noise points, which are labeled as -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')

# Plot the clustering result
plt.figure(figsize=(10, 6))

# Scatter plot of the data points, color-coded by cluster
unique_labels = set(labels)
for label in unique_labels:
    # Filter points by the cluster label
    class_member_mask = (labels == label)
    
    # Assign colors: black for noise, random colors for other clusters
    if label == -1:
        color = 'k'  # Noise points in black
    else:
        color = plt.cm.Spectral(float(label) / n_clusters)  # Assign colors to clusters
    
    plt.scatter(lon_data_all[class_member_mask], lat_data_all[class_member_mask], 
                c=[color], s=10, label=f'Cluster {label}' if label != -1 else 'Noise')

# Set plot titles and labels
plt.title(f'Clustering of Lightning Events (DBSCAN) - {n_clusters} Clusters Found', fontsize=15)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show the plot with a legend
plt.legend(loc='upper right')
plt.show()
