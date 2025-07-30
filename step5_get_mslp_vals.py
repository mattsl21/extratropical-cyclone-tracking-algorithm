#! /usr/bin/env python
##########################################################################################
#Program to  get the central pressure of an ETC
#Author: Matthew Lynne
##########################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
from datetime import datetime, timedelta
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

#############################################
#Load in SC file and smooth MSLP netCDF files
#Change filepaths based on user need
#############################################
centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', dtype = 'str')
centers_info = centers_info.to_numpy(dtype = 'str')

new_centers_info = {"Central Pressure (hPa)":[]}

mslp_5059 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_195001-195912smooth.nc').msl
mslp_6069 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_196001-196912smooth.nc').msl
mslp_7078 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197001-197812smooth.nc').msl
mslp_7988 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197901-198812smooth.nc').msl
mslp_8998 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_198901-199812smooth.nc').msl
mslp_9909 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_199901-200912smooth.nc').msl
mslp_1022 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_201001-202212smooth.nc').msl
mslp_2324 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_202301-202404smooth.nc').msl

for row in range(0, len(centers_info)):
	year = centers_info[row, 0]
	month = centers_info[row, 1]
	day = centers_info[row, 2]
	hour = centers_info[row, 3]

	lon = centers_info[row, 4]
	lat = centers_info[row, 5]
	
	time_slice = str(year) + '-' + str(f"{month:02}") + '-' + str(f"{day:02}") + '-' + str(f"{hour:02}")
		
	fpath = mslp_5059
	
	if(float(year) >= 1950 and float(year) <= 1959):
		fpath = mslp_5059
	elif(float(year) >= 1960 and float(year) <= 1969):
		fpath = mslp_6069
	elif(float(year) >= 1970 and float(year) <= 1978):
		fpath = mslp_7078
	elif(float(year) >= 1979 and float(year) <= 1988):
		fpath = mslp_7988
	elif(float(year) >= 1989 and float(year) <= 1998):
		fpath = mslp_8998
	elif(float(year) >= 1999 and float(year) <= 2009):
		fpath = mslp_9909
	elif(float(year) >= 2010 and float(year) <= 2022):
		fpath = mslp_1022
	else:
		fpath = mslp_2324

	slp_val = float(fpath.sel(longitude = lon, latitude = lat, time = time_slice))

	new_centers_info['Central Pressure (hPa)'].append(round(slp_val, 2))

	if(row != len(centers_info) - 1 and centers_info[row + 1, 1] != month):
		print(datetime(int(year), int(month), int(day), int(hour)))

centers_info = pd.DataFrame(centers_info, dtype = 'str')
new_centers_info = pd.DataFrame(new_centers_info, dtype = 'str')

output = pd.concat([centers_info, new_centers_info], ignore_index = True, axis = 1)
output.columns = ['Year', 'Month', 'Day', 'Hour', 'Longitude', 'Latitude', 'ETC #', 'Central Pressure (hPa)']
output.to_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', index=False)



