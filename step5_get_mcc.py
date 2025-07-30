#! /usr/bin/env python

################################################################
#Program to find all line numbers in sc_track_clean_0150_to_0424
#Corresponding to ETCs that may be comprised of a multi-centered
#cyelone
#Author: Matthew Lynne
################################################################
import numpy as np
import pandas as pd
import math 
from datetime import datetime, timedelta

##################################################
#Load in data. File path will differ based on user
##################################################
centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', dtype = 'str')
centers_info = centers_info.to_numpy(dtype = 'str')

new_centers_info = {"Year":[], "Month":[], "Day":[], "Hour":[], "Longitude":[], "Latitude":[]} 

mcc_possibilities = []

curr_time = datetime(1950, 1, 1, 0)
delta = timedelta(hours = 1)

########################################################
#Distance between 2 points (lon1, lat1) and (lon2, lat2)
########################################################

def distance(lon1, lat1, lon2, lat2):
        radius = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c
        return d

##################################################################
#Loop through all datetime objects until 01 October 2024 at 00 UTC
#This can be changed depending on time period of analysis
##################################################################
while(curr_time != datetime(2024, 10, 1, 0)):
	#List of lat/lon pairs at the current time step
	locs = []
	
	if(curr_time.day == 1 and curr_time.hour == 0):
		print(curr_time)
	
	year = str(curr_time.year)
	month = str(f"{curr_time.month:02}")
	day = str(f"{curr_time.day:02}")
	hour = str(f"{curr_time.hour:02}")
	
	#Next six lines identify all locations in sc_tracks_clean_0150_to_0424 corresponding to the current time
	yr_inds = np.where(centers_info[:, 0] == year)
	min_yrind = min(yr_inds[0])
	max_yrind = max(yr_inds[0]) + 1
	match_mask = np.logical_and(np.logical_and(np.logical_and((centers_info[min_yrind:max_yrind, 1] == month), \
	(centers_info[min_yrind:max_yrind, 2] == day)), (centers_info[min_yrind:max_yrind, 3] == hour)), centers_info[min_yrind:max_yrind, 0] == year)
	inds = np.where(match_mask) + min_yrind
	
	inds = inds[0].tolist()
	
	#Get the lon/lat of each index
	for ind in inds:
		locs.append((centers_info[ind, 4], centers_info[ind, 5]))
	
	index = 0
	#Loop through each location in locs
	for location in locs:
		loc_to_test = location
		
		#Look through the other locations in locs
		for temp_loc in locs:
			if(temp_loc != loc_to_test):
				lon1 = int(loc_to_test[0])
				lat1 = int(loc_to_test[1])
				lon2 = int(temp_loc[0])
				lat2 = int(temp_loc[1])

				#Two centers at the same time step must be less than 1000 km apart
				if(distance(lon1, lat1, lon2, lat2) < 1000):
					mcc_possibilities.append(inds[index])
		index = index + 1

	if(curr_time.month == 4 and curr_time.day == 30 and curr_time.hour == 23):
		curr_time = datetime(int(year), 10, 1, 0)
	else:
		curr_time = curr_time + delta

mcc_possibilities = np.array(mcc_possibilities)
mcc_possibilities = np.unique(mcc_possibilities)

#File path will change depending on user need
np.savetxt('../../../files/filtered_mslp/through_2023/mcc_poss.txt', mcc_possibilities, fmt = '%i')
