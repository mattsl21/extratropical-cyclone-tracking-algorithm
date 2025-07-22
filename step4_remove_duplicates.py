#! /usr/bin/env python
###########################################################################
#Program to remove duplicates that arose from step 3 of the algorithm
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
from joblib import Parallel, delayed

centers_info = pd.read_csv('../../../files/filtered_mslp/through_2023/sc_tracks_0150_to_0424', dtype = 'str')

new_centers_info = centers_info.drop_duplicates(subset = ['Year', 'Month', 'Day', 'Hour', 'Longitude', 'Latitude'], ignore_index = True)

new_centers_info.to_csv('../../../files/filtered_mslp/through_2023/sc_tracks_clean_0150_to_0424', index=False)
