from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

# Path to your NetCDF file
nc_file_path = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\Lightning_Project\pyltg\pyltg\examples\test_files\TRMM_LIS_SC.04.3_2015.087.98925.nc'

# Open the NetCDF file
nc_file = Dataset(nc_file_path, mode='r')

# Check the available variables
print("Variables in the file:", nc_file.variables.keys())

# Assuming 'raster_image' is a variable containing image data
raster_image_data = nc_file.variables.get('raster_image', None)

# If raster_image_data exists, plot it
if raster_image_data is not None:
    # Extract the data (assuming it's 2D or 3D; adjust based on actual dimensions)
    image_data = raster_image_data[:]

    # Plotting the image data
    plt.imshow(image_data, cmap='gray')  # You can change 'gray' to another colormap
    plt.colorbar()  # Add a colorbar for reference
    plt.title('Raster Image from NetCDF File')
    plt.show()
else:
    print("No 'raster_image' variable found in the file.")

# Close the file
nc_file.close()
