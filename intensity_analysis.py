import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where your NetCDF TRMM LIS files are stored
directory_path = "C:/Users/jbull/OneDrive - Fayetteville State University/CSC490 SENIOR PROJECT/Lightning_Project/pyltg/pyltg/examples/test_files/"

# List all NetCDF files in the directory
file_names = [file for file in os.listdir(directory_path) if file.endswith(".nc")]

# Initialize a list to store intensity data
intensity_data_list = []

# Loop over each file, open the dataset, and extract intensity data
for file in file_names:
    file_path = os.path.join(directory_path, file)
    
    # Open the NetCDF dataset
    try:
        dataset = nc.Dataset(file_path, mode='r')
        
        # Extract intensity variable (assuming 'lightning_event_radiance')
        intensity_data = dataset.variables.get('lightning_event_radiance', None)
        
        if intensity_data is not None:
            # Clean and append the intensity data (remove NaN values)
            intensity_data_clean = intensity_data[~np.isnan(intensity_data)]
            intensity_data_list.extend(intensity_data_clean)
        
        # Close the dataset after processing
        dataset.close()
        
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert intensity data to a numpy array for easier handling
intensity_data_all = np.array(intensity_data_list)

# Plot the intensity distribution
plt.figure(figsize=(10, 6))

# Create a histogram of the intensity data
plt.hist(intensity_data_all, bins=50, color='green', edgecolor='black')

# Set title and labels
plt.title('Distribution of Lightning Event Intensity (Radiance)', fontsize=15)
plt.xlabel('Intensity (Radiance)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plot
plt.show()
