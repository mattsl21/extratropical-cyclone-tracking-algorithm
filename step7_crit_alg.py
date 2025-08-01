#! /usr/bin/env python
##########################################################################################
#Program to find the SLP value of each SCC's outermost isobar following the criteria in 
#Lynne and Dai (2024)
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
import math
from scipy.ndimage.filters import minimum_filter, maximum_filter
from skimage import measure
from scipy.interpolate import NearestNDInterpolator
import time
from datetime import datetime, timedelta
from matplotlib.axes import Axes
from matplotlib.path import Path
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from joblib import Parallel, delayed

#Initial filepath will differ depending on user
centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', dtype = 'str')
centers_info = centers_info.to_numpy(dtype = 'str')

#Eventual output files for the list of vertices for each ETC and the row numbers corresponding to the sc_tracks_clean_0150_to_0424 line number
output = open('../../../files/filtered_mslp/through_2023/crits.txt', 'a')
rows = open('../../../files/filtered_mslp/through_2023/rows.txt', 'a')

new_centers_info = {"Row":[], "Critical MSLP (Pa)":[]}
	
vertices = []
old_vertices = []
real_etcs = []
isnan = 0

####################################################################################################
#Very similar to get_centers() method in Step 1, but edited to identify SLP maxima within a 3*3 grid
#See Step 1 for further commentary
####################################################################################################
def get_max_centers(slp_field):
        centers_lon_list = []
        centers_lat_list = []

        a = slp_field
         
        maxima = (a == maximum_filter(a, 3, mode='constant', cval=0.0))
  
        res = np.array(np.where(1 == maxima))
        
        lon_range = a.shape[1]
        
        lat_range = a.shape[0]
      
        lon_test_list = [0, 359]
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

                        if a[y, x] > a[y, x1] and a[y, x] > a[y, xm1] and \
                        a[y, x] > a[y1, x] and a[y, x] > a[y1, xm1] and \
                        a[y, x] > a[y1, x1] and a[y, x] > a[ym1, x] and \
                        a[y, x] > a[ym1, xm1] and a[y, x] > a[ym1, x1]:
                        #a[y, x] > a[y, x2] and a[y, x] > a[y, xm2] and \
                        #a[y, x] > a[y2, x] and a[y, x] > a[y2, xm2] and \
                        #a[y, x] > a[y2, x2] and a[y, x] > a[ym2, x] and \
                        #a[y, x] > a[ym2, xm2] and a[y, x] > a[ym2, x2] and \
                        #a[y, x] > a[y1, x2] and a[y, x] > a[y1, xm2] and \
                        #a[y, x] > a[ym1, x2] and a[y, x] > a[ym1, xm2] and \
                        #a[y, x] > a[y2, x1] and a[y, x] > a[y2, xm1] and \
                        #a[y, x] > a[ym2, xm1] and a[y, x] > a[ym2, x1]:

                                centers_lon_list.append(x)
                                centers_lat_list.append((90 - y))
        
	for index in range(0, len(res[0])):
                if(res[0, index] >= 12 and res[0, index] <= 70):
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
                        if a[y, x] > a[y, x1] and a[y, x] > a[y, xm1] and \
                        a[y, x] > a[y1, x] and a[y, x] > a[y1, xm1] and \
                        a[y, x] > a[y1, x1] and a[y, x] > a[ym1, x] and \
                        a[y, x] > a[ym1, xm1] and a[y, x] > a[ym1, x1]:
                        #a[y, x] > a[y, x2] and a[y, x] > a[y, xm2] and \
                        #a[y, x] > a[y2, x] and a[y, x] > a[y2, xm2] and \
                        #a[y, x] > a[y2, x2] and a[y, x] > a[ym2, x] and \
                        #a[y, x] > a[ym2, xm2] and a[y, x] > a[ym2, x2] and \
                        #a[y, x] > a[y1, x2] and a[y, x] > a[y1, xm2] and \
                        #a[y, x] > a[ym1, x2] and a[y, x] > a[ym1, xm2] and \
                        #a[y, x] > a[y2, x1] and a[y, x] > a[y2, xm1] and \
                        #a[y, x] > a[ym2, xm1] and a[y, x] > a[ym2, x1]:

                                centers_lon_list.append(x)
                                centers_lat_list.append((90 - y))
        return centers_lon_list, centers_lat_list


def get_full_path(list_of_paths):
	vts = []

	#Append each vertex point to vts
	for i in range(0, len(list_of_paths)):
		for v in list_of_paths[i].vertices:
			vts.append(v)
	
	#Have the end of vts be identical to the beginning
	vts.append(vts[0])
	vts = np.array(vts)
	ret_path = Path(vts)
	
	return ret_path

def check_criteria(center_lon, center_lat, fip_max, time, tr_paths, etc_arr):
	
	if(len(tr_paths) == 0):
		return False
	
	#This makes sure the first and last positions in the path are the same
	if(round(tr_paths[0].vertices[0, 1], 4) != round(tr_paths[-1].vertices[-1, 1], 4)):
		return False

	mins = []
	maxes = []
	
	#Make sure the path does not dip into the southern hemisphere or encircle the North Pole
	for line in tr_paths:
		if(np.any(line.vertices[:, 1] < 0) or np.any(np.around(line.vertices[:, 1], 4) == 90.0000)):
			return False
		minv = min(line.vertices[:, 0])
		maxv = max(line.vertices[:, 0])

		#More stringent requirement here than previously
		if((maxv - minv) > 358):
			return False

	#See if the path contains the ETC central lat/lon
	#elif statement checks if the longitude is zero, or at the edge of the domain
	#Note that if len(tr_paths) > 1, then the path crosses the dateline at least once
	if(len(tr_paths) == 1):

		if(tr_paths[0].contains_point((center_lon, center_lat)) == False):
			return False

	elif(center_lon == 0 and len(tr_paths) > 1):
		hasit = 0
		for p in tr_paths:
			if(0.0000 in np.around(p.vertices[:, 0], 4)):
				if(p.contains_point((0.01, center_lat)) == True):
					hasit = hasit + 1
			if(360.0000 in np.around(p.vertices[:, 0], 4)):
				if(p.contains_point((359.99, center_lat)) == True):
					hasit = hasit + 1
		if(hasit == 0):
			return False

	elif(len(tr_paths) > 1):
		hasit = 0
		for p in tr_paths:
			if(0.0000 in np.around(p.vertices[:, 0], 4)):
				if(p.contains_point((center_lon, center_lat)) == True):
					hasit = hasit + 1
			if(360.0000 in np.around(p.vertices[:, 0], 4)):
				if(p.contains_point((center_lon, center_lat)) == True):
					hasit = hasit + 1
		if(hasit == 0):
			return False

	#Separate paths into those with 0 and 360 in them separately
	have_e = False
	have_w = False
	verts_w = []
	verts_e = []
	for verts in range(0, len(tr_paths)):
		if(0.0000 in np.around(tr_paths[verts].vertices[:, 0], 4)):
			verts_e.append(tr_paths[verts])
			have_e = True
		elif(360.0000 in np.around(tr_paths[verts].vertices[:, 0], 4)):
			verts_w.append(tr_paths[verts])
			have_w = True
		else:
			verts_w.append(tr_paths[verts])
			have_w = True

	#Then append the above to pts separately
	pts = []
	if(have_w == True):
		pts.append(get_full_path(verts_w))
	if(have_e == True):
		pts.append(get_full_path(verts_e))

	point = (center_lon, center_lat)
	y = time.year
	m = time.month
	d = time.day
	h = time.hour
	
	pos_max_list = fip_max
	
	#If a SLP maximum is at a longitude of zero, change it to a longitude of 0.1 just to see if the path in question encircles it.
	if(0 in pos_max_list[:, 0]):
		zero_ind = np.where(pos_max_list[:, 0] == 0)[0][0]
		pos_max_list[zero_ind, 0] = 0.1
	
	max_count = 0
	
	#Check if the outermost isobar encompasses any other SLP maxima or other identified SCC. If it does, then return False.
	for pt in pts:
		truth_vals = pt.contains_points(pos_max_list)	
		truth_spots = np.where(truth_vals == True)[0]		
		max_count = max_count + len(truth_spots)
		
		if(max_count > 1):
			return False
	
		for real_etc in range(0, len(etc_arr)):
			if(pt.contains_point((etc_arr[real_etc, 0], etc_arr[real_etc, 1])) and (etc_arr[real_etc, 0], etc_arr[real_etc, 1]) != point):
				return False
	
	return True
	

def get_true_paths(path_list, path_to_test, lon_pos):
	
	#Vertex paths to skip over
	skip_paths = []
	
	#List representing the initial return list
	ret_list = []

	#Get first and last lat/lon position in path_to_test
	first_pos = (round(path_to_test.vertices[0, 0], 4), round(path_to_test.vertices[0, 1], 4))
	last_pos = (round(path_to_test.vertices[-1, 0], 4), round(path_to_test.vertices[-1, 1], 4))
	
	#If the first and last positions are equal to each other
	if(first_pos == last_pos):
		ret_list.append(path_to_test)
		return ret_list
	
	
	ret_list.append(path_to_test)	
	skip_paths.append(path_to_test)
	original_path = path_to_test
	
	curr_path = path_to_test
	to_original = False
	
	#Index for looping through path_list
	ind = 0
	count = 0
	
	#While we are not looking at the original path
	while(to_original == False):
		other_path = path_list[ind]	
		if(other_path != curr_path):

			#Get the first and last positions of the other path "test_pos"
			test_first_pos = (round(other_path.vertices[0, 0], 4), round(other_path.vertices[0, 1], 4))
			test_last_pos = (round(other_path.vertices[-1, 0], 4), round(other_path.vertices[-1, 1], 4))
			if((test_first_pos[0] == 0.0000 and test_last_pos[0] == 0.0000) or (test_first_pos[0] == 360.0000 and test_last_pos[0] == 360.0000)):
				
				#Check if any positions in the test path match positions in the other path
				if((test_first_pos[1] == first_pos[1] or test_first_pos[1] == last_pos[1]) or \
				(test_last_pos[1] == first_pos[1] or test_last_pos[1] == last_pos[1])):
					
					#Ensure the other path is not supposed to be skipped
					if((other_path in skip_paths) == False):
						count = 0
						ret_list.append(other_path)
						curr_path = other_path

						#The new first pos becomes test_first_pos, same with last_pos
						first_pos = test_first_pos
						last_pos = test_last_pos
						skip_paths.append(other_path)
		
		#Check if we are back to the original path
		if(count == len(path_list) - 1):
			to_original = True
	
		count = count + 1
		if(ind != len(path_list) - 1):
			ind = ind + 1
		else:
			ind = 0
	
	#This is the list that will be returned. Cleaned version of ret_list.
	clean_ret_list = []
	clean_ret_list.append(ret_list[0])
	ret_list = np.array(ret_list)
	first_coord = ret_list[0].vertices[0]
	last_coord = None

	#Proceed from 360 to 0 with the same latitude
	if(round(first_coord[0], 4) == 360.0000):
		last_coord = np.array((0.0000, round(first_coord[1], 4)))
	elif(round(first_coord[0], 4) == 0.0000):
		last_coord = np.array((360.0000, round(first_coord[1], 4)))
	temp_last_coord = ret_list[0].vertices[-1]
	
	count = 0
	
	#Look for the next vertex after the end of each element in ret_list
	while(np.any(temp_last_coord != last_coord)):
		count = count + 1
		
		if(round(temp_last_coord[0], 4) == 360.0000):
			look_for = np.array((0.0000, round(temp_last_coord[1], 4)))
		elif(round(temp_last_coord[0], 4) == 0.0000):
			look_for = np.array((360.0000, round(temp_last_coord[1], 4)))
		
		#Go to next element in ret_list
		ret_list = np.roll(ret_list, 1)
		if(np.all(np.around(ret_list[0].vertices[0], 4) == look_for)):
			count = 0
			clean_ret_list.append(ret_list[0])
			temp_last_coord = np.around(ret_list[0].vertices[-1], 4)
		elif(np.all(np.around(ret_list[0].vertices[-1], 4) == look_for)):
			count = 0
			temp_arr = Path(np.flipud(ret_list[0].vertices))
			clean_ret_list.append(temp_arr)
			temp_last_coord = np.around(temp_arr.vertices[-1], 4)
	
		if(count == len(ret_list)):
			break

	return clean_ret_list

def get_crit_slp(date, cent_lon, cent_lat, cent_pres, field, p_max_list, etcs):
	yr = date.year
	mo = date.month
	da = date.day
	ho = date.hour
	curr_lon = cent_lon
	curr_lat = cent_lat
	x = np.arange(0, 360)
	y = np.arange(90, -5, -1)

	field = np.concatenate((field, field[:, :1]), axis=1)
	
	x_wrapped = np.concatenate((x, [360.0]))
	
	#Start by assuming we have the outermost isobar vertex list, or path
	have_path = True
	
	#Factor by which to multiply the current ETC center SLP
	mult = 2

	#The turn variable is zero if a SLP contour a multiple of 2 higher than the SCC SLP center meets the requirements outlined in Lynne and Dai (2025)
	turn = 0
	
	#While we have a vertex path and turn != 8, e.g. we have not gone 4 hPa without finding an outermost isobar candidate
	while(turn != 8 and have_path == True):
		true_paths = []
		real_paths = []
		here = 0
		
		#Check the contour 4 hPa higher than the ETC central SLP
		contour_to_check = cent_pres + (200 * mult)
		
		cs = measure.find_contours(field[:95, :], contour_to_check)
		paths = []
		
		#Look through each vertex list returned in cs and append to the paths list
		for vlist in cs:
			vlist = np.fliplr(vlist)
			vlist[:, 1] = 90 - vlist[:, 1]
			paths.append(Path(vlist))
		
		#Loop through each path in the paths list
		for path in paths:
			
			#Exit loop if vertex path dips into southern hemisphere or hits the North Pole
			if(np.any(path.vertices[:, 1] < 0) or np.any(np.around(path.vertices[:, 1], 4) == 90.0000)):
				continue

			minv = min(path.vertices[:, 0])
			maxv = max(path.vertices[:, 0])
			
			#Exit loop if vertex path fully encompasses the Northern Hemisphere (or North Pole)
			if(maxv - minv >= 359):
				continue
			
			#Now that the first two criteria are met, append the path to true_paths
			true_paths.append(path)

			#If the path is at the end of the longitude domain, reconnect it to the other side
			if(360.0000 in np.around(path.vertices[:, 0], 4)):
				true_paths = get_true_paths(paths, path, 360)
			elif(0.0000 in np.around(path.vertices[:, 0], 4)):
				true_paths = get_true_paths(paths, path, 0)
			
			#Check the remaining criteria
			good = check_criteria(cent_lon, cent_lat, p_max_list, date, true_paths, etcs)
			
			#If a multiple of 4 hPa greater than ETC central SLP and criteria are met, append to real paths and go to the next multiple of 4
			#Otherwise, return the identified contour as the outermost isobar
			if(here == 0 and good == True): 
				here = 1
				real_paths.append(path)
				
				if(turn == 0):
					mult = mult + 2
				else:
					return contour_to_check
			
			true_paths = []
		
		#If at a multiple of 4
		if(turn == 8):

			#If mult never increased, then just return the central pressure as the outermost isobar is less than 0.5 hPa greater than the central
			#SLP
			if(mult == 0):
				return cent_pres
			
			#Otherwise return the SLP value 4 hPa lower 
			return contour_to_check - (200 * mult)

		#Otherwise, check the isobar 0.5 hPa lower than what was done previously
		elif(len(real_paths) == 0):
			mult = mult - 0.25
			turn = turn + 1
	
	#Return the SLP value 0.5 hPa lower
	return contour_to_check - 50

def run_prog(row):
	#List of already existing SCC lat/lon centers
	real_etcs = []
	
	year = centers_info[row, 0]
	month = centers_info[row, 1]
	day = centers_info[row, 2]
	hour = centers_info[row, 3]
	longitude = float(centers_info[row, 4])
	latitude = float(centers_info[row, 5])
	central_pres = float(centers_info[row, 7])

	dt = datetime(int(year), int(month), int(day), int(hour))
	time_slice = str(year) + '-' + str(f"{month:02}") + '-' + str(f"{day:02}") + '-' + str(f"{hour:02}")

	central_lon = longitude
	central_lat = latitude

	#Identify existing SCCs at the hour of analysis in next five lines
	yr_inds = np.where(centers_info[:, 0] == year)
	min_yrind = min(yr_inds[0])
	max_yrind = max(yr_inds[0]) + 1
	match_mask = np.logical_and(np.logical_and(np.logical_and((centers_info[min_yrind:max_yrind, 1] == month), (centers_info[min_yrind:max_yrind, 2] == day)), (centers_info[min_yrind:max_yrind, 3] == hour)), centers_info[min_yrind:max_yrind, 0] == year)
	inds = np.where(match_mask) + min_yrind
	
	inds = inds[0].tolist()

	#Loop through all identified SCCs at the estimating hour in inds
	for real in range(0, len(inds)):
		real_lo = centers_info[inds[real], 4]
		real_la = centers_info[inds[real], 5]
		real_etcs.append((int(real_lo), int(real_la)))

	real_etcs = np.array(real_etcs)

	#Get the proper MSLP file. This will likely change depending on how user stores MSLP netCDF files
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
	
	#Get the MSLP field at the time step of analysis
	field = np.array(fpath.sel(longitude = slice(0, 359), latitude = slice(90, -4), time = time_slice))
	
	#Deal with missing values
	mask = np.where(~np.isnan(field))
	interp = NearestNDInterpolator(np.transpose(mask), field[mask])
	field = interp(*np.indices(field.shape))
	
	#Search for any other maxima at the time step of analysis
	pos_max_list = get_max_centers(field)
	pos_max_list = np.reshape(np.ravel(np.array(pos_max_list)), [(len(pos_max_list[0])), 2], order = 'F')
		
	#Get the outermost isobar SLP value of the SCC with center at (center_lon, center_lat)
	crit_slp = round(get_crit_slp(dt, central_lon, central_lat, central_pres, field, pos_max_list, real_etcs), 2)
		
	real_etcs = []
	
	old_vertices.clear()
	vertices.clear()	
	
	return row, crit_slp

#Loop through sc_tracks_clean_0150_to_0424 in 10000 row-long blocks
strt = 0
nd = 10000

#While nd < length of the sc_tracks_clean_0150_to_0424 file. This number could vary depending on time periods analyzed
while(nd < 4071428):
	if(strt == 0):
		mslp_5059 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_195001-195912smooth.nc').msl
	elif(strt == 500000):
		mslp_6069 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_196001-196912smooth.nc').msl
	elif(strt == 600000):
		mslp_5059.close()
	elif(strt == 1000000):
		mslp_7078 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197001-197812smooth.nc').msl
	elif(strt == 1100000):
		mslp_6069.close()
	elif(strt == 1500000):
		mslp_7988 = xr.open_dataset('../../..//files/ETC_netcdf/msl_era5_197901-198812smooth.nc').msl
	elif(strt == 1600000):
		mslp_7078.close()
	elif(strt == 2100000):
		mslp_8998 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_198901-199812smooth.nc').msl
	elif(strt == 2200000):
		mslp_7988.close()
	elif(strt == 2600000):
		mslp_9909 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_199901-200912smooth.nc').msl
	elif(strt == 2700000):
		mslp_8998.close()
	elif(strt == 3200000):
		mslp_1022 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_201001-202212smooth.nc').msl
	elif(strt == 3300000):
		mslp_9909.close()
	elif(strt == 3900000):
		mslp_2324 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_202301-202404smooth.nc').msl
	elif(strt == 4000000):
		mslp_1022.close()
	
	#Parallel computing recommended, but not necessary 
	crit_list = Parallel(n_jobs = 6)(delayed(run_prog)(r) for r in range(strt, nd))

	for sys in crit_list:
		output.write(str(sys[0]) + ',' + str(sys[1]) + '\n')
		rows.write(str(sys[0]) + '\n')
	strt = strt + 10000
	nd = nd + 10000
	crit_list.clear()

crit_slp = Parallel(n_jobs = 6)(delayed(run_prog)(r) for r in range(4070000, 4071428))
for sys in crit_slp:
	output.write(str(sys[0]) + ',' + str(sys[1]) + '\n')
	rows.write(str(sys[0]) + '\n')					       





