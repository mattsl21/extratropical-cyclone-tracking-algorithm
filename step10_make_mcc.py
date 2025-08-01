#! /usr/bin/env python

import numpy as np
import pandas as pd
import math as m
import xarray as xr
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
from datetime import datetime, timedelta

#Initial file path will change based on user
centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', dtype = 'str')
centers_info = centers_info.to_numpy(dtype = 'str')

#Load in the mcc_poss txt file
mcc_poss = np.loadtxt('../../../files/filtered_mslp/through_2023/mcc_poss.txt')

#Eventual path for the MCC vertices
output_01 = open('../../../files/filtered_mslp/through_2023/mc_clean_verts_01.txt', 'a')

#Dictionary for final output file
new_centers_info = {"Year":[], "Month":[], "Day":[], "Hour":[], "Longitude":[], "Latitude":[], "Central Pressure (Pa)":[], "Critical SLP (Pa)":[], "Area (km^2)":[], "MCC?":[], "Centers":[], "SC Line":[], "ETC #":[], "Verts File Line":[]} 

def distance(lon1, lat1, lon2, lat2):
        radius = 6371
        dlat = m.radians(lat2 - lat1)
        dlon = m.radians(lon2 - lon1)
        a = m.sin(dlat/2) * m.sin(dlat/2) + m.cos(m.radians(lat1)) * m.cos(m.radians(lat2)) * m.sin(dlon/2) * m.sin(dlon/2)
        c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
        d = radius * c
        return d

#This method has been included in Steps 1, 7, and 8. See those files for further commentary
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

                                centers_lon_list.append(x)
                               
                                centers_lat_list.append((90 - y))
        return centers_lon_list, centers_lat_list

#Same method as in Step 9
def area(x, y):
        d_area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return d_area

#Same method as in Step 9
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

#See Step 8 for further commentary
def get_full_path(list_of_paths):
        vts = []
        for i in range(0, len(list_of_paths)):
                for v in list_of_paths[i].vertices:
                        vts.append(v)
        vts.append(vts[0])
        vts = np.array(vts)
        ret_path = Path(vts)
       
        return ret_path

#Append each individual vertex to coords_list. Same as in Step 8
def get_coords(lp):
        coords_list = []
        have_e = False
        have_w = False
        verts_w = []
        verts_e = []
        for verts in range(0, len(lp)):
                for vert in lp[verts].vertices:
                      coords_list.append(str(np.around(vert, 4)))
                      coords_list.append(',')
                     
                coords_list.append('\n')
                
        return coords_list

#Another check for connecting paths and checking for points when the outermost isobar hits the edge of the domain
def points_test(slp_list, id_paths):
        if(len(id_paths) == 0):
                return False
        
	for p in range(0, len(slp_list)):
                center_lon = slp_list[p][0]
                center_lat = slp_list[p][1]
                
		if(len(id_paths) == 1):
                        
			if(id_paths[0].contains_point((center_lon, center_lat)) == False):
                                return False
                
		elif(center_lon == 0 and len(id_paths) > 1):
                        hasit = 0
                        
			for p in id_paths:
                                if(0.0000 in np.around(p.vertices[:, 0], 4)):
                                        if(p.contains_point((0.01, center_lat)) == True):
                                               hasit = hasit + 1
                                if(360.0000 in np.around(p.vertices[:, 0], 4)):
                                        if(p.contains_point((359.99, center_lat)) == True):
                                               hasit = hasit + 1
                        
			if(hasit == 0):
                                return False

                elif(len(id_paths) > 1):
                        hasit = 0
                        for p in id_paths:
                                if(0.0000 in np.around(p.vertices[:, 0], 4)):
                                        if(p.contains_point((center_lon, center_lat)) == True):
                                              hasit = hasit + 1
                                if(360.0000 in np.around(p.vertices[:, 0], 4)):
                                        if(p.contains_point((center_lon, center_lat)) == True):
                                              hasit = hasit + 1
                        if(hasit == 0):
                                return False
        return True 

#Same method exists in Step 8
def check_criteria(center_lon, center_lat, fip_max, time, tr_paths, etc_arr, mcc_list):
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

                if((maxv - minv) > 358):
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
                
                for real_etc in range(0, len(etc_arr)):
                        if(pt.contains_point((etc_arr[real_etc, 0], etc_arr[real_etc, 1])) and (etc_arr[real_etc, 0], etc_arr[real_etc, 1]) != point):
                                return False

        return True

#Same method as in Step 8
def get_true_paths(path_list, path_to_test, lon_pos):
        skip_paths = []
        ret_list = []
        
	first_pos = (round(path_to_test.vertices[0, 0], 7), round(path_to_test.vertices[0, 1], 7))
        last_pos = (round(path_to_test.vertices[-1, 0], 7), round(path_to_test.vertices[-1, 1], 7))
        
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
                        test_first_pos = (round(other_path.vertices[0, 0], 7), round(other_path.vertices[0, 1], 7))
                        test_last_pos = (round(other_path.vertices[-1, 0], 7), round(other_path.vertices[-1, 1], 7))
                        if((test_first_pos[0] == 0.0000000 and test_last_pos[0] == 0.0000000) or (test_first_pos[0] == 360.0000000 and test_last_pos[0] == 360.0000000)):
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
        if(round(first_coord[0], 7) == 360.0000000):
                last_coord = np.array((0.0000000, round(first_coord[1], 7)))
        elif(round(first_coord[0], 7) == 0.0000000):
                last_coord = np.array((360.0000000, round(first_coord[1], 7)))
        temp_last_coord = ret_list[0].vertices[-1]
       
        count = 0
        
        while(np.any(temp_last_coord != last_coord)):
                count = count + 1
               
                if(round(temp_last_coord[0], 7) == 360.0000000):
                        look_for = np.array((0.0000, round(temp_last_coord[1], 7)))
                elif(round(temp_last_coord[0], 7) == 0.0000000):
                        look_for = np.array((360.0000000, round(temp_last_coord[1], 7)))
                
                ret_list = np.roll(ret_list, 1)
                if(np.all(np.around(ret_list[0].vertices[0], 7) == look_for)):
                        count = 0
                        clean_ret_list.append(ret_list[0])
                        temp_last_coord = np.around(ret_list[0].vertices[-1], 7)
                elif(np.all(np.around(ret_list[0].vertices[-1], 7) == look_for)):
                        count = 0
                        temp_arr = Path(np.flipud(ret_list[0].vertices))
                        clean_ret_list.append(temp_arr)
                        temp_last_coord = np.around(temp_arr.vertices[-1], 7)
                
                if(count == len(ret_list)):
                        break

        return clean_ret_list

#Similar to previous steps, but with functionality to work with MCCs
def get_crit_slp(date, cent_lon, cent_lat, crit_press, field, p_max_list, etcs, poss_centers):
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

        have_path = True
       
        mult = 2
        turn = 0
        all_true_paths = []
        while(turn != 8 and have_path == True):
                true_paths = []
                real_paths = []
                here = 0
                contour_to_check = crit_press + (200 * mult)
                
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
                        
                        #Check criteria with other possible centers excluded from etc_arr
                        good = check_criteria(cent_lon, cent_lat, p_max_list, date, true_paths, etcs, poss_centers)
                        if(good):
                                all_true_paths.append(true_paths)

                        if(here == 0 and good == True): #Once the correct path found, check through all criteria
                                here = 1
                                real_paths.append(path)

                                if(turn == 0):
                                        mult = mult + 2
                                else:
                                        return contour_to_check, all_true_paths[-1]
                        true_paths = []
                
                if(turn == 8):
                        if(mult == 0):
                                return cent_pres, []
                        return contour_to_check - (200 * mult), all_true_paths[-1]
                elif(len(real_paths) == 0):
                        mult = mult - 0.25
                        turn = turn + 1
        if(len(all_true_paths) == 0):
                return contour_to_check - 50, []
        
        return contour_to_check - 50, all_true_paths[-1]

#Indices to skip
skip_inds = []
delim = '-----------\n'
line_no = 0

#List for new vertices file
new_vf = []

#Line number for new vertices file
new_vf_line_no = 0

#Line number for new MCC file
new_file_line_no = 0

for strt in range(0, len(centers_info)):
	print('Row', strt)
	print('SC line #', line_no)
	
	#Location of SLP data will change based on user
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
		mslp_7988 = xr.open_dataset('../../../files/ETC_netcdf/msl_era5_197901-198812smooth.nc').msl
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
	
	if(strt == 0):
		vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_01.txt', 'r')
		vfs = vf.readlines()
	elif(strt == 1000000):
		vf.close()
		vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_12.txt', 'r')
		vfs = vf.readlines()
		line_no = 0
	elif(strt == 2000000):
		vf.close()
		vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_23.txt', 'r')
		vfs = vf.readlines()
		line_no = 0
	elif(strt == 3000000):
		vf.close()
		vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_34.txt', 'r')
		vfs = vf.readlines()
		line_no = 0
	elif(strt == 4000000):
		vf.close()
		vf = open('../../../files/filtered_mslp/through_2023/sc_clean_verts_45.txt', 'r')
		vfs = vf.readlines()
		line_no = 0

	#If we have a possible MCC and we do not have to skip the SCC at this index, then proceed
	if(((strt in mcc_poss) == True) and ((strt in skip_inds) == False)):
		
		#First make sure that we do not analyze this index at any other time
		skip_inds.append(strt)
		
		row = strt
		mcc_poss_inds = []
		slp_pos = []
		real_etcs = []
		
		year = centers_info[int(row), 0]
		month = centers_info[int(row), 1]
		day = centers_info[int(row), 2]
		hour = centers_info[int(row), 3]
		dt = datetime(int(year), int(month), int(day), int(hour))
		time_slice = str(year) + '-' + str(f"{month:02}") + '-' + str(f"{day:02}") + '-' + str(f"{hour:02}")

		central_lon = int(centers_info[int(row), 4])
		central_lat = int(centers_info[int(row), 5])
		central_press = float(centers_info[int(row), 6])

		#Locate all indices where the time is the same as 'row' and the locations are less than 1000km apart, note the lat/lon location,
		#Index, and central SLP
		yr_inds = np.where(centers_info[:, 0] == year)
		min_yrind = min(yr_inds[0])
		max_yrind = max(yr_inds[0]) + 1
		match_mask = np.logical_and(np.logical_and(np.logical_and((centers_info[min_yrind:max_yrind, 1] == month), \
		(centers_info[min_yrind:max_yrind, 2] == day)), (centers_info[min_yrind:max_yrind, 3] == hour)), centers_info[min_yrind:max_yrind, 0] == year)
		same_time_inds = np.where(match_mask) + min_yrind
		same_time_inds = same_time_inds[0].tolist()
	
		for ind in same_time_inds:
			if((ind in mcc_poss) and (ind >= strt)):
				poss_lon = int(centers_info[ind, 4])
				poss_lat = int(centers_info[ind, 5])
				poss_central_press = float(centers_info[ind, 6])
				if(distance(central_lon, central_lat, poss_lon, poss_lat) < 1000):
					mcc_poss_inds.append((ind, poss_central_press))
					slp_pos.append((poss_lon, poss_lat))
				else:
					real_lo = centers_info[ind, 4]
					real_la = centers_info[ind, 5]
					real_etcs.append((int(real_lo), int(real_la)))
			else:
				real_lo = centers_info[ind, 4]
				real_la = centers_info[ind, 5]
				real_etcs.append((int(real_lo), int(real_la)))
	
		real_etcs = np.array(real_etcs)
	
		#Order mcc_poss_inds from lowest SLP to highest SLP
		mcc_poss_inds = np.array(mcc_poss_inds)
		sorted_inds = mcc_poss_inds[np.argsort(mcc_poss_inds[:, 1])]

		#Get the SLP and area of ETC with lowest SLP
		lowest_slp_ind = int(sorted_inds[0, 0])
		starting_contour = float(centers_info[lowest_slp_ind, 7])
		starting_area = float(centers_info[lowest_slp_ind, 8])

		#Get field information
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

		#Get SLP maxima
		pos_max_list = get_max_centers(field)
		pos_max_list = np.reshape(np.ravel(np.array(pos_max_list)), [(len(pos_max_list[0])), 2], order = 'F')
		
		#MCC test fails, as no other ETCs within 1000 km
		if(len(slp_pos) <= 1):
			year = centers_info[strt, 0]
			month = centers_info[strt, 1]
			day = centers_info[strt, 2]
			hour = centers_info[strt, 3]
			longitude = centers_info[strt, 4]
			latitude = centers_info[strt, 5]
			central_pres = centers_info[strt, 7]
			crit_slp = centers_info[strt, 8]
			a = centers_info[strt, 9]
			mcc = "N"
			e_no = centers_info[strt, 6]

			new_centers_info["Year"].append(year)
			new_centers_info["Month"].append(month)
			new_centers_info["Day"].append(day)
			new_centers_info["Hour"].append(hour)
			new_centers_info["Longitude"].append(longitude)
			new_centers_info["Latitude"].append(latitude)
			new_centers_info["Central Pressure (Pa)"].append(central_pres)
			new_centers_info["Critical SLP (Pa)"].append(crit_slp)
			new_centers_info["Area (km^2)"].append(a)
			new_centers_info["MCC?"].append(mcc)
			new_centers_info["Centers"].append('nan')
			new_centers_info["SC Line"].append(strt)
			new_centers_info["ETC #"].append(e_no)
			new_centers_info["Verts File Line"].append(new_vf_line_no)

			if(vfs[line_no].split(',')[-2] != '-999'):
				
				#Until a delimiter is hit, append the new file line number and vertex list to new_vf
				while(vfs[line_no] != delim):
					new_vf.append(str(new_file_line_no) + ',' + ' ' + str(vfs[line_no]))
					new_vf_line_no = new_vf_line_no + 1
					line_no = line_no + 1
				
				new_vf.append(delim)
				new_vf_line_no = new_vf_line_no + 1
				line_no = line_no + 1
			else:
				new_vf.append(str(new_file_line_no) + ',' + ' ' + str(vfs[line_no]))
				new_vf_line_no = new_vf_line_no + 1
				line_no = line_no + 1
		
		#While other ETCs within 1000 km exist
		while(len(slp_pos) > 1):
			print(sorted_inds)

			#Find contour that meets criteria without other possible minima included in etc_arr
			crit_slp, crit_slp_path = get_crit_slp(dt, central_lon, central_lat, starting_contour, field, pos_max_list, real_etcs, sorted_inds)
			print(crit_slp)
		
			#Check if path returned above contains the other points
			has_points = points_test(slp_pos, crit_slp_path)
	
			#Calculate the area of crit_slp_path
			new_area = get_area(crit_slp_path)
			print(new_area)
	
			#Official MCC test
			if(has_points and starting_area/new_area < 0.5):
				print('yes')
				p_or_s = 'P'
				for i in sorted_inds[:, 0]:
					skip_inds.append(int(i))
				
				#Add to dictionary for case where an MCC is detected
				nind = int(sorted_inds[0, 0])
				nyear = centers_info[nind, 0]
				nmonth = centers_info[nind, 1]
				nday = centers_info[nind, 2]
				nhour = centers_info[nind, 3]
				nlon = centers_info[nind, 4]
				nlat = centers_info[nind, 5]
				ncentral = centers_info[nind, 7]
				ncrit = crit_slp
				narea = round(new_area, 1)
				nmcc = 'Y'
				ncenters = slp_pos
				ne_no = centers_info[nind, 6]

				new_centers_info["Year"].append(nyear)
				new_centers_info["Month"].append(nmonth)
				new_centers_info["Day"].append(nday)
				new_centers_info["Hour"].append(nhour)
				new_centers_info["Longitude"].append(nlon)
				new_centers_info["Latitude"].append(nlat)
				new_centers_info["Central Pressure (Pa)"].append(ncentral)
				new_centers_info["Critical SLP (Pa)"].append(ncrit)
				new_centers_info["Area (km^2)"].append(narea)
				new_centers_info["MCC?"].append(nmcc)
				new_centers_info["Centers"].append(ncenters)
				new_centers_info["SC Line"].append(nind)
				new_centers_info["ETC #"].append(ne_no)
				new_centers_info["Verts File Line"].append(new_vf_line_no)

				crds = get_coords(crit_slp_path)
				new_vf.append(str(new_file_line_no) + ' ' + str(new_file_line_no))
				
				nline_count = 0
				for sys in crds:
					new_vf.append(sys)
					if(sys == '\n'):
						nline_count = nline_count + 1
				
				new_vf_line_no = new_vf_line_no + nline_count
				new_vf.append(delim)
				new_vf_line_no = new_vf_line_no + 1
				if(vfs[line_no].split(',')[-2] != '-999'):
					while(vfs[line_no] != delim):
						line_no = line_no + 1
					line_no = line_no + 1
				else:
					line_no = line_no + 1

				break
			else:
				print('no')
				slp_pos = slp_pos[:-1]
				sorted_inds = sorted_inds[:-1]

				#Add to dictionary for case when MCC test fails. Only add info for current row
				if(len(slp_pos) == 1):
					year = centers_info[strt, 0]
					month = centers_info[strt, 1]
					day = centers_info[strt, 2]
					hour = centers_info[strt, 3]
					longitude = centers_info[strt, 4]
					latitude = centers_info[strt, 5]
					central_pres = centers_info[strt, 7]
					crit_slp = centers_info[strt, 8]
					a = centers_info[strt, 9]
					e_no = centers_info[strt, 6]
					mcc = "N"
				
					new_centers_info["Year"].append(year)
					new_centers_info["Month"].append(month)
					new_centers_info["Day"].append(day)
					new_centers_info["Hour"].append(hour)
					new_centers_info["Longitude"].append(longitude)
					new_centers_info["Latitude"].append(latitude)
					new_centers_info["Central Pressure (Pa)"].append(central_pres)
					new_centers_info["Critical SLP (Pa)"].append(crit_slp)
					new_centers_info["Area (km^2)"].append(a)
					new_centers_info["MCC?"].append(mcc)
					new_centers_info["Centers"].append('nan')
					new_centers_info["SC Line"].append(strt)
					new_centers_info["ETC #"].append(e_no)
					new_centers_info["Verts File Line"].append(new_vf_line_no)
					
					if(vfs[line_no].split(',')[-2] != '-999'):
						while(vfs[line_no] != delim):
							new_vf.append(str(new_file_line_no) + ',' + ' ' + str(vfs[line_no]))
							new_vf_line_no = new_vf_line_no + 1
							line_no = line_no + 1
						new_vf.append(delim)
						new_vf_line_no = new_vf_line_no + 1
						line_no = line_no + 1
					else:
						new_vf.append(str(new_file_line_no) + ',' + ' ' + str(vfs[line_no]))
						new_vf_line_no = new_vf_line_no + 1
						line_no = line_no + 1
		
		#Remove the current row from skip_inds to prevent skip_inds from becoming too large
		while(int(row) in skip_inds):
			skip_inds.remove(row)
		new_file_line_no = new_file_line_no + 1
	
	elif(strt in skip_inds):
		while(strt in skip_inds):
			skip_inds.remove(strt)
		if(vfs[line_no].split(',')[-2] != '-999'):
			while(vfs[line_no] != delim):
				line_no = line_no + 1
		line_no = line_no + 1
	else:
		#Add to dictionary for SCC
		year = centers_info[strt, 0]
		month = centers_info[strt, 1]
		day = centers_info[strt, 2]
		hour = centers_info[strt, 3]
		longitude = centers_info[strt, 4]
		latitude = centers_info[strt, 5]
		central_pres = centers_info[strt, 7]
		crit_slp = centers_info[strt, 8]
		a = centers_info[strt, 9]
		e_no = centers_info[strt, 6]
		mcc = "N"

		new_centers_info["Year"].append(year)
		new_centers_info["Month"].append(month)
		new_centers_info["Day"].append(day)
		new_centers_info["Hour"].append(hour)
		new_centers_info["Longitude"].append(longitude)
		new_centers_info["Latitude"].append(latitude)
		new_centers_info["Central Pressure (Pa)"].append(central_pres)
		new_centers_info["Critical SLP (Pa)"].append(crit_slp)
		new_centers_info["Area (km^2)"].append(a)
		new_centers_info["MCC?"].append(mcc)
		new_centers_info["Centers"].append('nan')
		new_centers_info["SC Line"].append(strt)
		new_centers_info["ETC #"].append(e_no)
		new_centers_info["Verts File Line"].append(new_vf_line_no)

		new_vf.append(str(new_file_line_no) + ',' + ' ')
		if(vfs[line_no].split(',')[-2] != '-999'):
			while(vfs[line_no] != delim):
				new_vf.append(str(vfs[line_no]))
				new_vf_line_no = new_vf_line_no + 1
				line_no = line_no + 1
			new_vf.append(delim)
			new_vf_line_no = new_vf_line_no + 1
			line_no = line_no + 1
		else:
			new_vf.append(str(new_file_line_no) + ',' + ' ' + str(vfs[line_no]))
			new_vf_line_no = new_vf_line_no + 1
			line_no = line_no + 1
		new_file_line_no = new_file_line_no + 1

	#Make dataframe and append to file every 1000 rows
	if(strt % 1000 == 0 and strt > 0):
		print('Adding to file')
		new_centers_info = pd.DataFrame(new_centers_info, dtype = 'str')
		if(strt == 1000):
			new_centers_info.to_csv('../../../files/filtered_mslp/through_2023/mc_tracks_clean_0150_to_0424', index=False, mode = 'a')
		else:
			new_centers_info.to_csv('../../../files/filtered_mslp/through_2023/mc_tracks_clean_0150_to_0424', index=False, header = False, mode = 'a')
		
		for el in new_vf:
			output_01.write(el)
		new_vf.clear()
		
		new_centers_info = {"Year":[], "Month":[], "Day":[], "Hour":[], "Longitude":[], "Latitude":[], "Central Pressure (Pa)":[], "Critical SLP (Pa)":[], "Area (km^2)":[], "MCC?":[], "Centers":[], "SC Line":[], "ETC #":[], "Verts File Line":[]}
	
	if(strt == len(centers_info) - 1):
		new_centers_info = pd.DataFrame(new_centers_info, dtype = 'str')
		new_centers_info.to_csv('../../../files/filtered_mslp/through_2023/mc_tracks_clean_0150_to_0424', index=False, header = False, mode = 'a')
		for el in new_vf:
			output_01.write(el)
