from netCDF4 import Dataset

# Path to your NetCDF file
nc_file_path = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\Lightning_Project\pyltg\pyltg\examples\test_files\TRMM_LIS_SC.04.3_2015.087.98925.nc'

# Open the NetCDF file
nc_file = Dataset(nc_file_path, mode='r')

# Extract variables that contain 'radiance'
radiance_variables = [var for var in nc_file.variables.keys() if 'radiance' in var]

# Print the variables that contain 'radiance'
print("Variables containing 'radiance':", radiance_variables)

# Access data from those variables
for var in radiance_variables:
    radiance_data = nc_file.variables[var][:]
    print(f"\nData for variable '{var}':")
    print(radiance_data)

# Close the file after use
nc_file.close()

