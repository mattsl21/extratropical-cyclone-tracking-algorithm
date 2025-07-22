#! /usr/bin/env python
#################################################################################################
#Program to track all identified MSLP centers
#MSLP centers must last for at least 24 hours and first and last separateion greater than 1000km
#Step 3 of ETC tracking algorithm
#Author: Matthew Lynne
#################################################################################################
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

#Can change file path based on user server setup
centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/filtered_grad')


centers_info = centers_info.to_numpy()
new_centers_info = {"Year":[], "Month":[], "Day":[], "Hour":[], "Longitude":[], "Latitude":[], "ETC #":[]}
tr_new = open('../../../files/filtered_mslp/through_2023/sc_tracks_new.txt', "w+")
dt_new = open('../../../files/filtered_mslp/through_2023/sc_datetimes_new.txt', "w+")
id_new = open('../../../files/filtered_mslp/through_2023/sc_indices_new.txt', "w+")

#########################
#Initialize lists
#########################
lon_track = [] 
lat_track = [] 
skip_indices = [] 
track_indices = []
dates = []

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

##################################################
#Append to CSV file given track number and indices
##################################################
def make_csv(index_list, t_no):
	ind_list = []
	
	for ind in index_list:
		ind_list.append(len(new_centers_info['Year']))
		new_centers_info['Year'].append(centers_info[ind, 0])
		ayear = centers_info[ind, 0]
		new_centers_info['Month'].append(f"{int(centers_info[ind, 1]):02}")
		new_centers_info['Day'].append(f"{int(centers_info[ind, 2]):02}")
		new_centers_info['Hour'].append(f"{int(centers_info[ind, 3]):02}")
		new_centers_info['Longitude'].append(centers_info[ind, 4])
		new_centers_info['Latitude'].append(centers_info[ind, 5])
		new_centers_info['ETC #'].append(t_no)
		
		if(ind == index_list[-1]):
			print(date(ayear, centers_info[ind, 1], centers_info[ind, 2]))
	
	id_new.write(str(ind_list) + '\n')

###########################################################################
#Given an initial time, lat, and lon, track the MSLP center forward in time
###########################################################################
def check_forward_steps(index, time, lon, lat):
	delta = timedelta(hours = 1)
	ntime = time
	
	#Get to first index of next timestep
	while(ntime == time):

		#Check if end of centers_info has been reached
		if(index == len(centers_info) - 1):
			return
		else:
			index = index + 1
		
		nyear = int(centers_info[index, 0])
		nmonth = int(centers_info[index, 1])
		nday = int(centers_info[index, 2])
		nhour = int(centers_info[index, 3])
		nlon = int(centers_info[index, 4])
		nlat = int(centers_info[index, 5])
		ntime = datetime(nyear, nmonth, nday, nhour)

	curr_time = ntime
	time_mult = 1
	still_exists = True

	#Check each identified center at the next time step
	#If the system still exists and the difference between the current time and original time is a multiple of 1
	while(still_exists and curr_time - time == timedelta(hours = 1*time_mult)):
		
		#While in a certain timestep
		still_exists = False
		count = 0
		
		#Potential lons, lats, skip indices, and track indices
		pot_lons = []
		pot_lats = []
		pot_skip_inds = []
		pot_track_inds = []
		
		while(ntime == curr_time):
			#Check if the new lon and lat are within 500 km of what is already in lon_track and lat_track
			if(distance(lon_track[-1], lat_track[-1], nlon, nlat) < 500):
				
				pot_lons.append(nlon)
				pot_lats.append(nlat)
				
				pot_skip_inds.append(index)
				pot_track_inds.append(index)
				
				still_exists = True
				
				lon = nlon
				lat = nlat
				
				count = count + 1

			#Check if at the end of centers_info
			if(index == len(centers_info) - 1):
				return
			else:
				index = index + 1
			
			#Now go to the next index
			nyear = int(centers_info[index, 0])
			nmonth = int(centers_info[index, 1])
			nday = int(centers_info[index, 2])
			nhour = int(centers_info[index, 3])
			nlon = int(centers_info[index, 4])
			nlat = int(centers_info[index, 5])
			ntime = datetime(nyear, nmonth, nday, nhour)

		#Now, check all of the potential centers at this timestep and append closest one to the official track
		distances = []
		for pot_lon, pot_lat in zip(pot_lons, pot_lats):
			distances.append(distance(lon_track[-1], lat_track[-1], pot_lon, pot_lat))
		
		distances_arr = np.array(distances)

		if(len(distances_arr > 0)):
			min_pos = np.where(distances_arr == np.min(distances_arr))[0]
			lon_track.append(pot_lons[min_pos[0]])
			lat_track.append(pot_lats[min_pos[0]])
			skip_indices.append(pot_skip_inds[min_pos[0]])
			track_indices.append(pot_skip_inds[min_pos[0]])
			dates.append(curr_time)
		
		#Set the current time as the next time step
		curr_time = ntime
		time_mult = time_mult + 1	
	return

############################################################################
#Given an initial time, lat, and lon, track the MSLP center backward in time
############################################################################
def check_backward_steps(index, time, lon, lat):
        
        delta = timedelta(hours = -1)
        ntime = time

        #Get to index of previous timestep
        while(ntime == time):
                if(index == 0):
                        return
                else:
                        index = index - 1
                nyear = int(centers_info[index, 0])
                nmonth = int(centers_info[index, 1])
                nday = int(centers_info[index, 2])
                nhour = int(centers_info[index, 3])
                nlon = int(centers_info[index, 4])
                nlat = int(centers_info[index, 5])
                ntime = datetime(nyear, nmonth, nday, nhour)
        
        curr_time = ntime
        time_mult = 1
        still_exists = True

        #Check each identified system at the previous time step
        #If the system still exists and the difference between the current time and original time is a multiple of 1
        while(still_exists and curr_time - time == timedelta(hours = -1*time_mult)):

                #While in a certain timestep
                still_exists = False
                count = 0
                
		#Potential lons, lats, skip indices, and track indices
		pot_lons = []
                pot_lats = []
                pot_skip_inds = []
                pot_track_inds = []

                while(ntime == curr_time):
                        
                        if(distance(lon_track[0], lat_track[0], nlon, nlat) < 500):
                               
                                pot_lons.append(nlon)
                                pot_lats.append(nlat)
                                
                                pot_skip_inds.append(index)
                                pot_track_inds.append(index)
                                
				still_exists = True
                                lon = nlon
                                lat = nlat
                                count = count + 1

                        if(index == 0):
                                return
                        else:
                                index = index - 1
                        nyear = int(centers_info[index, 0])
                        nmonth = int(centers_info[index, 1])
                        nday = int(centers_info[index, 2])
                        nhour = int(centers_info[index, 3])
                        nlon = int(centers_info[index, 4])
                        nlat = int(centers_info[index, 5])
                        ntime = datetime(nyear, nmonth, nday, nhour)

                #Now, check all of the potential centers at this timestep and append closest one to the official track
                distances = []
                for pot_lon, pot_lat in zip(pot_lons, pot_lats):
                        distances.append(distance(lon_track[0], lat_track[0], pot_lon, pot_lat))
                distances_arr = np.array(distances)
  
                if(len(distances_arr > 0)):
                        min_pos = np.where(distances_arr == np.min(distances_arr))[0]
                        lon_track.insert(0, pot_lons[min_pos[0]])
                        lat_track.insert(0, pot_lats[min_pos[0]])
                        
                        track_indices.insert(0, pot_skip_inds[min_pos[0]])
                        dates.insert(0, curr_time)
                
		#Set the current time as the next time step
                curr_time = ntime
                time_mult = time_mult + 1
        return


track_no = 0
tot_count = 0
remaining_count = 0
line_no = 1

#Loop through each row in centers_info
for row in range(0, len(centers_info)):

	#Some rows may already be part of other SCCs and can be skipped, otherwise continue on
	if((row in skip_indices) == False):
		year = int(centers_info[row, 0])
		month = int(centers_info[row, 1])
		day = int(centers_info[row, 2])
		hour = int(centers_info[row, 3])
		longitude = int(centers_info[row, 4])
		latitude = int(centers_info[row, 5])

		dt = datetime(year, month, day, hour)
		
		#Append initial longitude, latitude, and row number to their respective lists
		lon_track.append(longitude)
		lat_track.append(latitude)
		track_indices.append(row)
		dates.append(dt)
		
		#Check the time steps after the initial time step
		check_forward_steps(row, dt, longitude, latitude)
		
		#Check the time steps before the initial time step to see if minimum can be tracked backwards in time
		check_backward_steps(row, dt, longitude, latitude)
		
		#Lasts longer than 24 hours
		if(len(lon_track) > 23):
			lon1 = lon_track[0]
			lat1 = lat_track[0]
			lon2 = lon_track[-1]
			lat2 = lat_track[-1]
			#Travels at least 1000 km
			if(distance(lon1, lat1, lon2, lat2) >= 1000):
				tot_count = tot_count + 1
		
				remaining_count = remaining_count + 1
				
				lon_track = np.array(lon_track).reshape(1, len(lon_track))
				lat_track = np.array(lat_track).reshape(1, len(lat_track))
				pos_track = np.concatenate((lon_track, lat_track), axis = 1).reshape((len(lon_track[0]), 2), order = 'F').tolist()
				
				#Write the track positions to tr_new
				tr_new.write(str(pos_track) + '\n') 
				true_dates = [d.strftime("%Y:%m:%d:%H") for d in dates]
				
				dt_new.write(str(true_dates) + '\n')
			
				lon_track = lon_track.tolist()
				lat_track = lat_track.tolist()
				line_no = line_no + 1
				line_no = line_no + 1
				line_no = line_no + 1
				
				#Make the CSV file with the SCC centers that meet the specified criteria
				make_csv(track_indices, track_no)
				track_no = track_no + 1
		
		lon_track.clear()
		lat_track.clear()
		
		track_indices.clear()
		dates.clear()
	else:
		while(row in skip_indices):
			skip_indices.remove(row)
	
df = pd.DataFrame(new_centers_info)

#The file path can be changed based on user server setup
df.to_csv('../../../files/filtered_mslp/through_2023/sc_tracks_0150_to_0424', index=False)

