#! /usr/bin/env python
#################################################################################################
#Program to txt files containing the vertices for each outermost isobar for each ETC at each hour
#Author: Matthew Lynne
#################################################################################################
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

#Filepath to open with ETC data that will likely change for each user
centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', dtype = 'str')
centers_info = centers_info.to_numpy(dtype = 'str')

#Eventual filepaths to store the ETC vertices
#01 corresponds to lines 0 to 1 million in sc_tracks_clean_0150_to_0424
#12 corresponds to lines 1 to 2 million in sc_tracks_clean_0150_to_0424
#Same for each of the following files
output_01 = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_01.txt', 'a')
output_12 = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_12.txt', 'a')
output_23 = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_23.txt', 'a')
output_34 = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_34.txt', 'a')
output_45 = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_45.txt', 'a')

#File to output the rows that have been analyzed
#This file is not completely necessary but can help keep track of algorithm progress
rows = open('../../../files/filtered_mslp/through_2023/rows.txt', 'a')

new_centers_info = {"Row":[], "Critical MSLP (Pa)":[]}

vertices = []
old_vertices = []
real_etcs = []

isnan = 0

#Same method exists in Step 7. See that program and Step 1 for further commentary
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

#Same method exists in Step 7. See Step 7 for further commentary.
def get_full_path(list_of_paths):
	vts = []
	
	for i in range(0, len(list_of_paths)):
		for v in list_of_paths[i].vertices:
			vts.append(v)
	
	vts.append(vts[0])
	vts = np.array(vts)
	ret_path = Path(vts)
	
	return ret_path

def get_coords(lp):
        coords_list = []
        
	#Following four lines are no longer needed
	have_e = False
        have_w = False
        verts_w = []
        verts_e = []

	#Loop through the list of paths and append each vertex to coords_list
        for verts in range(0, len(lp)):
                for vert in lp[verts].vertices:
                      coords_list.append(str(np.around(vert, 4)))
                      coords_list.append(',')
                      
                coords_list.append('\n')
        
	#Append a line to separate each SCC when writing the file
        coords_list.append('-----------')
        coords_list.append('\n')
       
        return coords_list

#This same method exists in Step 7. See Step 7 for detailed commentary
def check_criteria(center_lon, center_lat, fip_max, time, tr_paths):
	if(len(tr_paths) == 0):
		return False
	
	if(round(tr_paths[0].vertices[0, 1], 4) != round(tr_paths[-1].vertices[-1, 1], 4)):
		return False

	mins = []
	maxes = []

	for line in tr_paths:
		if(np.any(line.vertices[:, 1] < 0) or np.any(np.around(line.vertices[:, 1], 4) == 90.0000)):
			return False
		minv = min(line.vertices[:, 0])
		maxv = max(line.vertices[:, 0])

		if(maxv - minv > 358):
			return False

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
	if(0 in pos_max_list[:, 0]):
		zero_ind = np.where(pos_max_list[:, 0] == 0)[0][0]
		pos_max_list[zero_ind, 0] = 0.1
	max_count = 0
	for pt in pts:
		truth_vals = pt.contains_points(pos_max_list)
		truth_spots = np.where(truth_vals == True)[0]
		max_count = max_count + len(truth_spots)
		if(max_count > 1):
			return False
		
		for real_etc in range(0, len(real_etcs)):
			if(pt.contains_point((real_etcs[real_etc, 0], real_etcs[real_etc, 1])) and (real_etcs[real_etc, 0], real_etcs[real_etc, 1]) != point):
				return False
	return True
	
#Same program occurs in Step 7. See Step 7 for further commentary
def get_true_paths(path_list, path_to_test, lon_pos):
	skip_paths = []
	
	ret_list = []
	
	first_pos = (round(path_to_test.vertices[0, 0], 4), round(path_to_test.vertices[0, 1], 4))
	last_pos = (round(path_to_test.vertices[-1, 0], 4), round(path_to_test.vertices[-1, 1], 4))
	
	if(first_pos == last_pos):
		ret_list.append(path_to_test)
		return ret_list

	ret_list.append(path_to_test)	
	skip_paths.append(path_to_test)
	original_path = path_to_test
	curr_path = path_to_test
	to_original = False
	ind = 0
	count = 0
	
	while(to_original == False):
		other_path = path_list[ind]	
		
		if(other_path != curr_path):
			test_first_pos = (round(other_path.vertices[0, 0], 4), round(other_path.vertices[0, 1], 4))
			test_last_pos = (round(other_path.vertices[-1, 0], 4), round(other_path.vertices[-1, 1], 4))
			
			if((test_first_pos[0] == 0.0000 and test_last_pos[0] == 0.0000) or (test_first_pos[0] == 360.0000 and test_last_pos[0] == 360.0000)):
				
				if((test_first_pos[1] == first_pos[1] or test_first_pos[1] == last_pos[1]) or \
				(test_last_pos[1] == first_pos[1] or test_last_pos[1] == last_pos[1])):
					
					if((other_path in skip_paths) == False):
						count = 0
						ret_list.append(other_path)
						curr_path = other_path
						first_pos = test_first_pos
						last_pos = test_last_pos
						skip_paths.append(other_path)
		
		if(count == len(path_list) - 1):
			to_original = True
		
		count = count + 1
		
		if(ind != len(path_list) - 1):
			ind = ind + 1
		else:
			ind = 0
	
	
	clean_ret_list = []
	clean_ret_list.append(ret_list[0])
	ret_list = np.array(ret_list)
	first_coord = ret_list[0].vertices[0]
	last_coord = None
	
	if(round(first_coord[0], 4) == 360.0000):
		last_coord = np.array((0.0000, round(first_coord[1], 4)))
	elif(round(first_coord[0], 4) == 0.0000):
		last_coord = np.array((360.0000, round(first_coord[1], 4)))
	
	temp_last_coord = ret_list[0].vertices[-1]

	count = 0
	
	while(np.any(temp_last_coord != last_coord)):
		count = count + 1
		
		if(round(temp_last_coord[0], 4) == 360.0000):
			look_for = np.array((0.0000, round(temp_last_coord[1], 4)))
		elif(round(temp_last_coord[0], 4) == 0.0000):
			look_for = np.array((360.0000, round(temp_last_coord[1], 4)))
		
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

#Similar to get_crit_slp() method in Step 7
#Key difference is that if the SLP value of the outermost isobar is less than 0.5 hPa higher than the central SLP
#Then a value of -999 will be written into the eventual return file
#Furthermore, no need to check multiple SLP values for the outermost isobar, therefore only one SLP value is checked
def get_crit_slp(date, cent_lon, cent_lat, cent_pres, cr_slp, field, p_max_list):
	yr = date.year
	mo = date.month
	da = date.day
	ho = date.hour
	curr_lon = cent_lon
	curr_lat = cent_lat
	x = np.arange(0, 360)
	y = np.arange(90, -91, -1)
	field = np.concatenate((field, field[:, :1]), axis=1)
	x_wrapped = np.concatenate((x, [360.0]))

	have_path = True
	
	mult = 2
	turn = 0
	true_paths = []
	real_paths = []
	here = 0
	contour_to_check = cr_slp
	cs = measure.find_contours(field[:95, :], contour_to_check)
	paths = []
	
	for vlist in cs:
		vlist = np.fliplr(vlist)
		vlist[:, 1] = 90 - vlist[:, 1]
		paths.append(Path(vlist))

	for path in paths:
		if(np.any(path.vertices[:, 1] < 0) or np.any(np.around(path.vertices[:, 1], 4) == 90.0000)):
			continue
		minv = min(path.vertices[:, 0])
		maxv = max(path.vertices[:, 0])
		if(maxv - minv >= 359):
			continue

		
		true_paths.append(path)
		if(360.0000 in np.around(path.vertices[:, 0], 4)):
			true_paths = get_true_paths(paths, path, 360)
		elif(0.0000 in np.around(path.vertices[:, 0], 4)):
			true_paths = get_true_paths(paths, path, 0)

		good = check_criteria(cent_lon, cent_lat, p_max_list, date, true_paths)
		
		if(good == True):
			real_paths.append(true_paths)
		true_paths = []
	
	if(len(real_paths) > 1):
		vert_tots = []
		for plist in real_paths:
			vert_tot = 0
			if(len(plist) == 1):
				vert_tot = plist[0].vertices.shape[0]
				vert_tots.append(vert_tot)
			else:
				for pah in plist:
					vert_tot = vert_tot + pah.vertices.shape[0]
				vert_tots.append(vert_tot)
		vert_tots = np.array(vert_tots)
		m_ind = np.where(vert_tots == min(vert_tots))[0]
		if(m_ind.shape[0] > 1):
			m_ind = m_ind[0]
		else:
			m_ind = m_ind[0]
		
		list_to_ret = get_coords(real_paths[m_ind])
		return list_to_ret
	elif(len(real_paths) == 0):
		
		list_to_ret = ['-999', ',', '\n']
		return list_to_ret
	else:
		
		list_to_ret = get_coords(real_paths[0])
		return list_to_ret

row_count = []
def run_prog(row):
                print(row)
                real_etcs = []
                year = centers_info[row, 0]
                month = centers_info[row, 1]
                day = centers_info[row, 2]
                hour = centers_info[row, 3]
                longitude = float(centers_info[row, 4])
                latitude = float(centers_info[row, 5])
                central_pres = float(centers_info[row, 7])
                crit = float(centers_info[row, 8])
                if(central_pres == crit):
                        return row, ['-999', ',', '\n']
                dt = datetime(int(year), int(month), int(day), int(hour))
              
                time_slice = str(year) + '-' + str(f"{month:02}") + '-' + str(f"{day:02}") + '-' + str(f"{hour:02}")

                central_lon = longitude
                central_lat = latitude

                yr_inds = np.where(centers_info[:, 0] == year)
                min_yrind = min(yr_inds[0])
                max_yrind = max(yr_inds[0]) + 1

                match_mask = np.logical_and(np.logical_and(np.logical_and((centers_info[min_yrind:max_yrind, 1] == month), (centers_info[min_yrind:max_yrind, 2] == day)), (centers_info[min_yrind:max_yrind, 3] == hour)), centers_info[min_yrind:max_yrind, 0] == year)
                inds = np.where(match_mask) + min_yrind
                inds = inds[0].tolist()

                for real in range(0, len(inds)):
                        real_lo = centers_info[inds[real], 4]
                        real_la = centers_info[inds[real], 5]
                        real_etcs.append((int(real_lo), int(real_la)))
                real_etcs = np.array(real_etcs)

                #Depending on how the user stored SLP files, this block of code will likely change
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

                field = np.array(fpath.sel(longitude = slice(0, 359), latitude = slice(90, -4), time = time_slice))
                
		#Deal with missing values
                mask = np.where(~np.isnan(field))
                interp = NearestNDInterpolator(np.transpose(mask), field[mask])
                field = interp(*np.indices(field.shape))

                pos_max_list = get_max_centers(field)
                pos_max_list = np.reshape(np.ravel(np.array(pos_max_list)), [(len(pos_max_list[0])), 2], order = 'F')
                
		#Get the coordinates of the official outermost isobar
                crds = get_crit_slp(dt, central_lon, central_lat, central_pres, crit, field, pos_max_list)
                
                real_etcs = []

                old_vertices.clear()
                vertices.clear()
                row_count.append(row)
                
		return row, crds

#Similar to Step 7, loop through sc_tracks_clean_0150_to_0424 every 10000 rows
strt = 0
nd = 10000

#These blocks will also likely change depending on how user stores SLP data
while(nd < 4071428):
        if(strt == 0):
               mslp_5059 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_195001-195912smooth.nc').msl
        elif(strt == 500000):
               mslp_6069 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_196001-196912smooth.nc').msl
        elif(strt == 600000):
               mslp_5059.close()
        elif(strt == 1000000):
               output_01.close()
               mslp_7078 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197001-197812smooth.nc').msl
        elif(strt == 1100000):
               mslp_6069.close()
        elif(strt == 1500000):
               mslp_7988 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197901-198812smooth.nc').msl
        elif(strt == 1600000):
                mslp_7078.close()
        elif(strt == 2100000):
                output_12.close()
                mslp_8998 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_198901-199812smooth.nc').msl
        elif(strt == 2200000):
                mslp_7988.close()
        elif(strt == 2600000):
                mslp_9909 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_199901-200912smooth.nc').msl
        elif(strt == 2700000):
                mslp_8998.close()
        elif(strt == 3200000):
                output_23.close()
                mslp_1022 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_201001-202212smooth.nc').msl
        elif(strt == 3300000):
                mslp_9909.close()
        elif(strt == 3900000):
                mslp_2324 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_202301-202404smooth.nc').msl
        elif(strt == 4000000):
                mslp_1022.close()

        crit_list = Parallel(n_jobs = 6)(delayed(run_prog)(r) for r in range(strt, nd))
        
	#Write the coordinates to the proper output file
        for sys in crit_list:
              rows.write(str(sys[0]) + '\n')
              if(int(sys[0]) < 1000000):
                     output_01.write(str(sys[0]) + ',')
                     for el in sys[1]:
                          output_01.write(el)		     
              elif(int(sys[0]) < 2000000 and int(sys[0]) >= 1000000):
                     output_12.write(str(sys[0]) + ',')
                     for el in sys[1]:
                          output_12.write(el)		     
              elif(int(sys[0]) < 3000000 and int(sys[0]) >= 2000000):
                     output_23.write(str(sys[0]) + ',')
                     for el in sys[1]:
                          output_23.write(el)
              elif(int(sys[0]) < 4000000 and int(sys[0]) >= 3000000):   
                     output_34.write(str(sys[0]) + ',')
                     for el in sys[1]:
                          output_34.write(el)
              elif(int(sys[0]) < 5000000 and int(sys[0]) >= 4000000):
                     output_45.write(str(sys[0]) + ',')
                     for el in sys[1]:
                          output_45.write(el)		     
  
        strt = strt + 10000
        nd = nd + 10000
        
        crit_list.clear()

mslp_2324 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_202301-202404smooth.nc').msl

#Loop through the final lines of sc_tracks_clean_0150_to_0424 in which nd is not a multiple of 10000
crit_slp = Parallel(n_jobs = 6)(delayed(run_prog)(r) for r in range(4070000, 4071428))
for sys in crit_slp:
	rows.write(str(sys[0]) + '\n')
	output_45.write(str(sys[0]) + ',')
	for el in sys[1]:
		output_45.write(el)

	



