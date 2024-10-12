import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Directory where your NetCDF TRMM LIS files are stored
directory_path = "C:/Users/jbull/OneDrive - Fayetteville State University/CSC490 SENIOR PROJECT/Lightning_Project/pyltg/pyltg/examples/test_files/"

# List all NetCDF files in the directory
file_names = [file for file in os.listdir(directory_path) if file.endswith(".nc")]

# Initialize a list to store time data
time_data_list = []

# Function to convert TAI93 time to datetime
def convert_tai93_to_datetime(tai93_time):
    base_time = datetime(1993, 1, 1) + timedelta(seconds=tai93_time)
    return base_time

# Loop over each file, open the dataset, and extract time data
for file in file_names:
    file_path = os.path.join(directory_path, file)
    
    # Open the NetCDF dataset
    try:
        dataset = nc.Dataset(file_path, mode='r')
        
        # Extract time variable (assuming 'lightning_event_TAI93_time')
        time_data = dataset.variables.get('lightning_event_TAI93_time', None)
        
        if time_data is not None:
            # Convert TAI93 time to datetime
            time_data_clean = time_data[~np.isnan(time_data)]
            time_data_converted = [convert_tai93_to_datetime(t) for t in time_data_clean]
            
            # Append the time data to the list
            time_data_list.extend(time_data_converted)
        
        # Close the dataset after processing
        dataset.close()
        
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Plot the frequency of lightning events over time
plt.figure(figsize=(10, 6))

# Create a histogram of lightning events over time
plt.hist(time_data_list, bins=50, color='blue', edgecolor='black')

# Set title and labels
plt.title('Temporal Distribution of Lightning Events', fontsize=15)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Number of Events', fontsize=12)

# Show the plot
plt.show()
