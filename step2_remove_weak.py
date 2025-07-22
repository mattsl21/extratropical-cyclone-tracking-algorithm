#! /usr/bin/env python
###########################################################################
#Program to check if identified MSLP centers meet 7.5 hPa criteria
#Average of four grid cells 1000 km N,S,E,W must be 7.5 hPa greater than center
#This serves as step 2 of the tracking algorithm
#Author: Matthew Lynne
##########################################################################3
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
from datetime import date, datetime, timedelta
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/nhem_hrly_mslp_centers_01_50_to_04_24') #Chaange as needed
centers_info = centers_info.to_numpy()

new_centers_info = {"Year":[], "Month":[], "Day":[], "Hour":[], "Longitude":[], "Latitude":[]} #Set up new dict

#########################################################
#Program to calculate distance between two lat/lon points
#########################################################
def distance(lon1, lat1, lon2, lat2):
        radius = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c
        return d

#####################################################################################
#Program to get a new lat/lon given an original and a certain distance and angle away
#####################################################################################
def get_new_pos(lon, lat, dist, phi):
        rad = 6371
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)

        phi_rad = math.radians(phi)
        dist_rad = dist/rad

        new_lat = math.asin(math.sin(lat_rad) * math.cos(dist_rad) + math.cos(lat_rad) * math.sin(dist_rad) * math.cos(phi_rad))
        new_lon = lon_rad + math.atan2(math.sin(phi_rad) * math.sin(dist_rad) * math.cos(lat_rad), math.cos(dist_rad) - math.sin(lat_rad) * math.sin(new_lat))

        new_lon_deg = math.degrees(new_lon)
        new_lat_deg = math.degrees(new_lat)

        return new_lon_deg, new_lat_deg

here = 0
field = None
change_list = ['1959-12-31-23', '1969-12-31-23', '1978-12-31-23', '1988-12-31-23', '1998-12-31-23', '2009-12-31-23', '2022-12-31-23']
for row in range(0, len(centers_info)):
	year = int(centers_info[row, 0])
	
	#This if block was used by the author based on MSLP data stored on server. This can be changed for individual users
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
		
	month = f"{centers_info[row, 1]:02}"
	day = f"{centers_info[row, 2]:02}"
	hour = f"{centers_info[row, 3]:02}"
	dt = datetime(year, int(month), int(day), int(hour))
	longitude = int(centers_info[row, 4])
	latitude = int(centers_info[row, 5])
	#print(longitude, latitude)
	#print(longitude - 360, latitude)
	
	arr_latitude = 90 - latitude
	time_slice = str(year) + '-' + str(month) + '-' + str(day) + '-' + str(hour)
	
	if(time_slice in change_list):
		here = 0
	
	if(row != 0 and dt != prev_dt):
		print(prev_dt)
		field = np.array(mslp_00Z.sel(longitude = slice(0, 360), latitude = slice(90, 10), time = time_slice))
	elif(row == 0):
		field = np.array(mslp_00Z.sel(longitude = slice(0, 360), latitude = slice(90, 10), time = time_slice))
	
	tr_field = field

	#Get center point and locations N,S,E,W
	cp = tr_field[arr_latitude, longitude]
	north = get_new_pos(longitude, latitude, 1000, 0)
	east = get_new_pos(longitude, latitude, 1000, 90)
	south = get_new_pos(longitude, latitude, 1000, 180)
	west = get_new_pos(longitude, latitude, 1000, 270)

	pos_list = [north, east, south, west]
	grad_list = []
	
	count = 0
	for elem in pos_list:
		pos_list[count] = (int(round(elem[0], 0)), int(round(elem[1], 0)))
		
		#Account for situations in which the point is at the longitude 'ends' of the domain
		if(elem[0] < 0):
			pos_list[count] = (int(360 - abs(round(elem[0], 0))), int(round(elem[1], 0)))
		elif(elem[0] > 359):
			pos_list[count] = (int(round(elem[0], 0) - 360), int(round(elem[1], 0)))

		if(pos_list[count][0] == 360):
			pos_list[count] = (0, pos_list[count][1])
		count = count + 1
	
	#Append the differences in MSLP to grad_list
	for elem in pos_list:
		grad_list.append(abs(cp - tr_field[90 - elem[1], elem[0]]))
	
	mean_grad = np.mean(np.array(grad_list))

	#Only append to new_centers_info if the mean gradient is greater than 7.5 hPa or 750 Pa
	if(mean_grad > 750):
		#print('pass')
		new_centers_info["Year"].append(year)
		new_centers_info["Month"].append(month)
		new_centers_info["Day"].append(f"{day:02}")
		new_centers_info["Hour"].append(hour)
		new_centers_info["Longitude"].append(longitude)
		new_centers_info["Latitude"].append(latitude)	

	prev_dt = datetime(year, int(month), int(day), int(hour))

df = pd.DataFrame(new_centers_info)

#Can also change the following line based on user need
df.to_csv('../../../files/filtered_mslp/through_2023/filtered_grad', index=False)

