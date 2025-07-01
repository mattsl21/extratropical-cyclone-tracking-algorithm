#! /usr/bin/env python

#Step one of the Lynne and Dai (2025) ETC tracking algorithm
#Identify all MSLP minima given netCDF files of previously smoothed MSLP fields
#Author: Matthew Lynne

######################
#Import modules
######################
import numpy as np
import xarray as xr
import pandas as pd
import csv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage.filters import minimum_filter
from datetime import datetime

#########################################################
#Initialize the dictionary to store the MSLP minima data
#########################################################
centers_info = {"Year":[], "Month":[], "Day":[], "Hour":[], "Longitude":[], "Latitude":[]} 

#########################################################
#Algorithm to take in a MSLP field and locate all minima
#########################################################
def get_centers(slp_field):
	centers_lon_list = []
	centers_lat_list = []

	a = slp_field
	
	#Next two lines: get locations of all minima within a 3*3 grid box
	#Goal is to reduce the number of lat/lon combinations to loop through
	minima = (a == minimum_filter(a, 3, mode='constant', cval=0.0))
	
	res = np.array(np.where(1 == minima))

	lon_range = a.shape[1]

	lat_range = a.shape[0]

	lon_test_list = [0, 359]	
	
	#Loop through all latitudes between 80 and 20degN and all longitudes
	#Check if they are a minimum within a 5*5 box
	for y in np.arange(2, lat_range - 2):
		for x in lon_test_list:
			if(x == 359):
				x1 = 0
				x2 = 1
				xm1 = 358
				xm2 = 357
			elif(x == 358):
				x1 = 359
				x2 = 0
				xm1 = 357
				xm2 = 356
			elif(x == 0):
				x1 = 1
				x2 = 2
				xm1 = 359
				xm2 = 358
			elif(x == 1):
				x1 = 2
				x2 = 3
				xm1 = 0
				xm2 = 359
			else:
				x1 = x + 1
				x2 = x + 2
				xm1 = x - 1
				xm2 = x -2
			
			y1 = y + 1
			y2 = y + 2
			ym1 = y - 1
			ym2 = y - 2

			if a[y, x] < a[y, x1] and a[y, x] < a[y, xm1] and \
			a[y, x] < a[y1, x] and a[y, x] < a[y1, xm1] and \
			a[y, x] < a[y1, x1] and a[y, x] < a[ym1, x] and \
			a[y, x] < a[ym1, xm1] and a[y, x] < a[ym1, x1] and \
			a[y, x] < a[y, x2] and a[y, x] < a[y, xm2] and \
			a[y, x] < a[y2, x] and a[y, x] < a[y2, xm2] and \
			a[y, x] < a[y2, x2] and a[y, x] < a[ym2, x] and \
			a[y, x] < a[ym2, xm2] and a[y, x] < a[ym2, x2] and \
			a[y, x] < a[y1, x2] and a[y, x] < a[y1, xm2] and \
			a[y, x] < a[ym1, x2] and a[y, x] < a[ym1, xm2] and \
			a[y, x] < a[y2, x1] and a[y, x] < a[y2, xm1] and \
			a[y, x] < a[ym2, xm1] and a[y, x] < a[ym2, x1]:
						
				centers_lon_list.append(x)
				centers_lat_list.append((82 - y))
	
	#Now check all previously identified 3*3deg minima and see if they are also 5*5deg minima
	#This looks at possible minima along the edges of the domain
	for index in range(0, len(res[0])):
		if(res[0, index] >= 2 and res[0, index] <= 60):
			x = res[1, index]
			y = res[0, index]
			if(x == 359):
				x1 = 0
				x2 = 1
				xm1 = 358
				xm2 = 357
			elif(x == 358):
				x1 = 359
				x2 = 0
				xm1 = 357
				xm2 = 356
			elif(x == 0):
				x1 = 1
				x2 = 2
				xm1 = 359
				xm2 = 358
			elif(x == 1):
				x1 = 2
				x2 = 3
				xm1 = 0
				xm2 = 359
			else:
				x1 = x + 1
				x2 = x + 2
				xm1 = x - 1
				xm2 = x -2
			
			y1 = y + 1
			y2 = y + 2
			ym1 = y - 1
			ym2 = y - 2

			if a[y, x] < a[y, x1] and a[y, x] < a[y, xm1] and \
			a[y, x] < a[y1, x] and a[y, x] < a[y1, xm1] and \
			a[y, x] < a[y1, x1] and a[y, x] < a[ym1, x] and \
			a[y, x] < a[ym1, xm1] and a[y, x] < a[ym1, x1] and \
			a[y, x] < a[y, x2] and a[y, x] < a[y, xm2] and \
			a[y, x] < a[y2, x] and a[y, x] < a[y2, xm2] and \
			a[y, x] < a[y2, x2] and a[y, x] < a[ym2, x] and \
			a[y, x] < a[ym2, xm2] and a[y, x] < a[ym2, x2] and \
			a[y, x] < a[y1, x2] and a[y, x] < a[y1, xm2] and \
			a[y, x] < a[ym1, x2] and a[y, x] < a[ym1, xm2] and \
			a[y, x] < a[y2, x1] and a[y, x] < a[y2, xm1] and \
			a[y, x] < a[ym2, xm1] and a[y, x] < a[ym2, x1]:
                                
				centers_lon_list.append(x)
				centers_lat_list.append((82 - y))
	
	return centers_lon_list, centers_lat_list          				

month_list = ['01', '02', '03', '04', '10', '11', '12']

change_list = ['1959-12-31-23', '1969-12-31-23', '1978-12-31-23', '1988-12-31-23', '1998-12-31-23', '2009-12-31-23', '2022-12-31-23']

year_list = [*range(1950, 2025)]

hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

here = 0

#Loop through each year
for year in year_list:

	#Open new files when necessary
	#Directory names and locations will need to be changed
	if(here == 0):
		if(year <= 1959):
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_195001-195912smooth.nc').msl
			here = 1
		elif(year > 1959 and year <= 1969):
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_196001-196912smooth.nc').msl
			here = 1
		elif(year > 1969 and year <= 1978):
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197001-197812smooth.nc').msl
			here = 1
		elif(year > 1978 and year <= 1988):
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197901-198812smooth.nc').msl
			here = 1
		elif(year > 1988 and year <= 1998):
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_198901-199812smooth.nc').msl
			here = 1
		elif(year > 1998 and year <= 2009):
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_199901-200912smooth.nc').msl
			here = 1
		elif(year > 2009 and year <= 2022):
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_201001-202212smooth.nc').msl
			here = 1
		else:
			mslp_00Z = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_202301-202404smooth.nc').msl
			here = 1
	if(year == 2024):
		month_list = ['01', '02', '03', '04']

	#Loop through each month
	for month in month_list:
		day_list = []
		if(month == '01' or month == '03' or month == '10' or month == '12'):
			day_list = range(1, 32)
		elif(month == '02' and year%4 == 0):
			day_list = range(1, 30)
		elif(month == '02' and year %4 != 0):
			day_list = range(1, 29)
		else:
			day_list = range(1, 31)
		
		#Loop through each day
		for day in day_list:
			date_time = datetime(int(year), int(month), int(day))
			print(date_time)
			
			#Loop through each hour
			for hour in hours:
				time_slice = str(year) + '-' + str(month) + '-' + str(day) + '-' + str(hour)
				
				if(time_slice in change_list):
					here = 0
				
				pos = mslp_00Z.sel(longitude = slice(0, 359), latitude = slice(82, 18), time = time_slice)

				data = np.array(pos)
				
				pos_list = get_centers(data)

				lon_list = pos_list[0] #Longitudes of MSLP minima
				lat_list = pos_list[1] #Latitudes of MSLP minima

				list_to_add = []
				
				#Add information to dictionary
				for val in range(len(lon_list)):
					centers_info["Year"].append(year)
					centers_info["Month"].append(month)
					centers_info["Day"].append(f"{day:02}")
					centers_info["Hour"].append(hour)
					centers_info["Longitude"].append(lon_list[val])
					centers_info["Latitude"].append(lat_list[val])
					
field_names = ['Year', 'Month', 'Day', 'Hour', 'Longitude', 'Latitude']

df = pd.DataFrame(centers_info)

#Save file to wherever necessary
df.to_csv('../../../files/filtered_mslp/through_2023/nhem_hrly_mslp_centers_01_50_to_04_24', index=False)

