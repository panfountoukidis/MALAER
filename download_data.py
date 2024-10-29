#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:31:17 2024

@author: pfountou
"""

import glob
import coda
import harp
import functools
import numpy as np
import os
import pandas as pd
import sys
import datetime as dtt
import xarray as xr
from pystac import Collection
import requests
import hashlib
from pystac_client import ItemSearch

#### Import functions from the 'utility_functions' script ####
sys.path.append('/home/p/pfountou/')
import utility_functions as tools


## Define the main output path(s) ##
main_outpath = '/mnt/lapaero_b/groups/lap_aero/Panagiotis_PhD/Data/'


#%% Download AERONET data

## Define some parameters ##
yr_start = 2018; yr_end = 2023 # define the start and end year
if yr_start == yr_end:
    yr_range = f'{yr_start}'
else:
    yr_range = '-'.join([f'{yr_start}', f'{yr_end}'])

par = 'AOD' # define the parameter that you want to download

# path_out = tools.make_folder(main_outpath+'AERONET/', f'{par}/')

## AERONET code names for the stations ##
stations = ['Thessaloniki', 'ATHENS-NOA', 'Finokalia-FKL'] # 'ATHENS_NTUA'

## For no-inversion parameters ##
url_val = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3'
inv_val = 'no'; par_list = [f'{par}20', 10]

## For inversion parametrs ##
# url_val = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3'
# inv_val = 'yes'; par_list = [f'{par}', 10, 'ALM20']

# for st in stations:
    
#     tools.download_AERONET(url_val, st, f'{yr_start}0101', f'{yr_end}1231', '%Y%m%d',
#                             par_list, inv_val, path_out+f'{st}_{par}_{yr_range}.txt')


#%% Download Copernicus CAMS ERA-5 Reanalysis data

## Define some parameters ##
stations = ['Thessaloniki', 'Athens', 'Finokalia', 'Patra', 'Volos']
st_coordinates = [[40.64, 22.93], [37.95, 23.74], [35.32, 25.66],
                  [38.23, 21.74], [39.36, 22.95]]

stations = ['Athens']; st_coordinates = [[37.95, 23.74]]

coordinates_dict = dict(zip(stations, st_coordinates))

# key_val = '176406:4159f14c-1d33-4e3f-9759-5e993ab07c69' # initial
key_val = 'e2efa42e-ad54-4740-82f2-aeb9d3089539'

yr_start = 2021; yr_end = 2021 # define by hand the start and end year
if yr_start == yr_end:
    yr_range = f'{yr_start}'
else:
    yr_range = '-'.join([f'{yr_start}', f'{yr_end}'])

## Define the time and the pressure info ##
yr_list = [f'{i}' for i in range(yr_start, yr_end+1)]
month_list = [f'{i:02d}' for i in range(11, 13)] # 1-13
day_list = [f'{i:02d}' for i in range(1, 32)] # 1-32
time_list = [f'{i:02d}:00' for i in range(24)] # 24
# press_list = ['600', '650', '700', '750', '775', '800', '825', '850', '875',
#               '900', '925', '950', '975', '1000']

press_list = ['150', '175', '200', '225', '250', '300', '350', '400', '450',
              '500', '550', '600', '650', '700', '750', '775', '800', '825',
              '850', '875', '900', '925', '950', '975', '1000']

## All (fixed) pressure values provided in CAMS (in hPa) ##
# press_list = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100',
#               '125', '150', '175', '200', '225', '250', '300', '350', '400',
#               '450', '500', '550', '600', '650', '700', '750', '775', '800',
#               '825', '850', '875', '900', '925', '950', '975', '1000']

data_info = [yr_list, month_list, day_list, time_list, yr_range, press_list]

## Define the parameters that you want to download and their reanalysis type ##
par_list = ['boundary_layer_height', 'relative_humidity', 'specific_humidity']
re_type = ['reanalysis-era5-single-levels', 'reanalysis-era5-pressure-levels',
            'reanalysis-era5-pressure-levels']

par_list = ['relative_humidity'] # 'boundary_layer_height',
re_type = ['reanalysis-era5-pressure-levels'] # 'reanalysis-era5-single-levels',

# par_list = ['specific_humidity']; re_type = ['reanalysis-era5-pressure-levels']

data_info_dict = dict(zip(par_list, re_type))
dcoor = 0.35 # the +/- range of a specific latitude and longitude values


## Test parameters ##

# press_list = ['900', '925', '950', '975', '1000']
# data_info = [yr_list, month_list, day_list, time_list, yr_range, press_list]
# par_list = ['boundary_layer_height']
# re_type = ['reanalysis-era5-single-levels']
# data_info_dict = dict(zip(par_list, re_type))

## --------------- ##


path_out = tools.make_folder(main_outpath, 'CAMS_ERA5_Reanalysis/')

for st in stations:
    
    st_lat, st_lon = coordinates_dict[st]
    coords_range = tools.area_of_interest(st_lat, st_lon, dcoor)
    
    print(f'{st} is running...')
    
    tools.download_ERA5(key_val, data_info_dict, data_info, coords_range,
                        st, path_out) # +f'{st}_{yr_range}'


#%% Download S5P L2 PAL data

stations = ['Thessaloniki', 'Athens', 'Finokalia', 'Patra', 'Volos']
st_coordinates = [[40.64, 22.93], [37.95, 23.74], [35.32, 25.66],
                  [38.23, 21.74], [39.36, 22.95]]
coordinates_dict = dict(zip(stations, st_coordinates))
st_lat, st_lon = coordinates_dict[stations[0]]
dcoor = 0.2
area_vals = tools.area_of_interest(st_lat, st_lon, dcoor)

outpath = '/mnt/lapaero_b/groups/lap_aero/Panagiotis_PhD/Tests/Inputs/'
dates = ['2021-01-01', '2021-01-02']

# tools.download_S5P_L2_PAL('L2__TCWV__', dates, area_vals, outpath)

