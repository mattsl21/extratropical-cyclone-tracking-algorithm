#! /usr/bin/env python
##########################################################################################
#Program to calculate the size of all SCCs
#Author: Matthew Lynne
##########################################################################################
import sys
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math as m
from scipy.ndimage.filters import minimum_filter
import time
from datetime import datetime, timedelta
from matplotlib.axes import Axes
from matplotlib.path import Path
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

#As in other steps, initial file path will differ based on user
centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', dtype = 'str')
centers_info = centers_info.to_numpy(dtype = 'str')

#Eventual output files for the SCC areas
output = open('../../../files/filtered_mslp/through_2023/areas.txt', 'a')
rows = open('../../../files/filtered_mslp/through_2023/area_rows.txt', 'a')


def area(x, y):
	d_area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
	return d_area 

def get_area(real_path):
	areas = []
	for p in real_path:
		longitudes = p.vertices[:, 0]
		latitudes = p.vertices[:, 1]
		rad = 6371
		lat_dist = m.pi * rad / 180.0
		y = [l * lat_dist for l in latitudes]
		x = [lo * lat_dist * m.cos(m.radians(l)) for lo, l in zip(longitudes, latitudes)]
		areas.append(area(x, y))
	
	areas = np.array(areas)
	true_area = np.sum(areas)
		
	return true_area

def add_lon(l):
	if(l == 359):
		return 0
	else:
		return l + 1

def get_grid_area(lon, lat):
	upper_left = (lon, lat)
	added_lon = add_lon(lon)
	upper_right = (added_lon, lat)
	lower_right = (added_lon, lat - 1)
	lower_left = (lon, lat - 1)
	vs = [upper_left, upper_right, lower_right, lower_left]
	vs = np.array(vs)
	grid_path = Path(vs)
	a = get_area([grid_path])
	return a

def get_proper_paths(list_of_verts):
	ret_list = []
	count = 0
	
	indicator = 0
	
	#Two lists if path goes into both hemispheres
	verts_1 = []
	verts_2 = []

	#Loop through the list of verts and append each individual point to the list
	for ufpath in list_of_verts:
		for position in ufpath:
			new_pos = position.replace('[', '').replace(']', '').split(' ')
			while('' in new_pos):
				new_pos.remove('')
			if(indicator == 0):
				verts_1.append(np.array(new_pos, dtype = 'float'))
			elif(indicator == 1):
				verts_2.append(np.array(new_pos, dtype = 'float'))
		if(indicator == 0):
			indicator = 1
		else:
			indicator = 0
	
	if(len(list_of_verts) == 1):
		path1 = Path(verts_1)
		ret_list.append(path1)
	else:
		path1 = Path(verts_1)
		path2 = Path(verts_2)
		ret_list.append(path1)
		ret_list.append(path2)
	
	return ret_list

delim = '-----------\n'

vert_list = []

line_no = 0

#Loop through each SCC
for row in range(0, len(centers_info)):	
		if(row%1000 == 0):
			print(row)

		#Open the proper file with outermost isobar vertices
		if(row == 0):
			vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_01.txt', 'r')
			vfs = vf.readlines()
		elif(row == 1000000):
			#vf.close()
			vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_12.txt', 'r')
			vfs = vf.readlines()
			line_no = 0
		elif(row == 2000000):
			vf.close()
			vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_23.txt', 'r')
			vfs = vf.readlines()
			line_no = 0
		elif(row == 3000000):
			vf.close()
			vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_34.txt', 'r')
			vfs = vf.readlines()
			line_no = 0
		elif(row == 4000000):
			vf.close()
			vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_45.txt', 'r')
			vfs = vf.readlines()
			line_no = 0
		
		vert_list = []
		longitude = int(centers_info[row, 4])
		latitude = int(centers_info[row, 5])
		central = centers_info[row, 7]
		cr = centers_info[row, 8]

		#Collect vertices for a system at a certain timestep
		#First, see if the outermost SLP is less than 0.5 hPa higher than the central SLP
		#In this case, the grid cell area of the central location is taken as the ETC area
		if(vfs[line_no].split(',')[1] == '-999'):
			line_no = line_no + 1
			ar = get_grid_area(longitude, latitude)
	
			output.write(str(round(ar, 1)) + '\n')
			rows.write(str(row) + '\n')
			continue
		
		#Go to the next SCC
		while(vfs[line_no] != delim):
			vert_list.append(vfs[line_no].replace('\n', '').split(',')[1:-1])
			line_no = line_no + 1
	
		proper_paths = get_proper_paths(vert_list)
	
		total_area = get_area(proper_paths)
	
		line_no = line_no + 1
		output.write(str(round(total_area, 1)) + '\n')
		rows.write(str(row) + '\n')


