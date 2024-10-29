#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:32:47 2024

@author: pfountou
"""

'''
--------------------------------------------------------------------------
This script is used to make overpass files from the S5P/TROPOMI satellite.
--------------------------------------------------------------------------
'''

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
import netCDF4 as nc

#### Import functions from the 'utility_functions' script ####
sys.path.append('/home/p/pfountou/')
import utility_functions as tools


## Define the path(s) ##
main_outpath = '/mnt/lapaero_b/groups/lap_aero/Panagiotis_PhD/Data/S5P-TROPOMI/'+\
    'Overpass_Files/'

## Define some initial parameters ##

# area(s) of interest and their coordinates #
stations = ['Thessaloniki', 'Athens', 'Finokalia', 'Patra', 'Volos']
st_coordinates = [[40.64, 22.93], [37.95, 23.74], [35.32, 25.66],
                  [38.23, 21.74], [39.36, 22.95]]

# stations = ['Thessaloniki']; st_coordinates = [[40.64, 22.93]]

coordinates_dict = dict(zip(stations, st_coordinates))

# other parameters #
qa_val, point_val, bands = 50, 20, [3] # irr_wls = 'UVN'
# error_mess = 'Error!! Something is wrong!!!' # , color_code = '\033[0;31;47m'
sat_products = ['RA_OFFL', 'IR_OFFL', 'TO3_NRTI', 'CLOUD_OFFL', 'CH4_OFFL', 'TCWV_PAL']
sat_products = ['CLOUD_OFFL']
d_type = 'overpass'
yr = '2021' # '2*'

for product_id in sat_products:

    ## Make the average/overpass files, depending the product ##
        
    if product_id in ['RA_OFFL', 'IR_OFFL']:
          
        for band in bands:
            
            ## Define by hand the wavelength band for the IR product ##
            if band == 3:
                wl = 'UVN'
            
            if product_id == 'RA_OFFL':
                data_path = f'RA_BD{band}/'
                product_file_nm = product_id+f'_BD{band}'
            
            else:
                data_path = f'IR_{wl}/'
                product_file_nm = product_id+f'_{wl}'
            
            main_inpath = '/mnt/lapaero_b/groups/lap_aero/Panagiotis_PhD/Data/'+\
                'S5P-TROPOMI/'+data_path
            
            ## Make the output folder ##
            # path_out = tools.make_folder(main_outpath, data_path)
            
            ## Get the files ##
            files = glob.glob(main_inpath+f'{yr}/*'); files.sort()
            # files = files[:10] # For testing!!
            
            ## Get the year range from the folder names ##
            year_folders = glob.glob(main_inpath+f'{yr}/'); year_folders.sort()
            year_range = tools.year_range(year_folders, '/')
            
            if product_id == 'IR_OFFL':
                
                file_name = '_'.join(['S5P', product_file_nm, year_range])
                input_list = [band]
                
                print(f'{product_id}: Running...')
                
                ## Initialize the average Harp product, just in case ##
                average_product = 0
                
                ## Get the average/overpass data ##
                average_product = tools.s5p_avrg_ovps(files, input_list, product_id,
                                                      d_type)
    
                ## Save the file ##
                ## try:
                
                # file_out = path_out+file_name
                # harp.export_product(average_product, file_out+'.nc')
                
                print('DONE!!') # {product_id} for {st_nm}:
                
                ## except: # harp.Error
                ##     print(f'{st_nm}: {color_code}{error_mess}')
                    ## pass
            
            else:
                
                for st_nm in stations[:1]:
                    
                    file_name = '_'.join(['S5P', product_file_nm, st_nm, 'Overpass',
                                          year_range])
                    
                    st_lat, st_lon = coordinates_dict[st_nm]
                    input_list = [st_lat, st_lon, point_val, band]
                    
                    print(f'{product_id} for {st_nm}: Running...')
                    
                    ## Initialize the average Harp product, just in case ##
                    average_product = 0
                    
                    ## Get the average/overpass data ##
                    average_product = tools.s5p_avrg_ovps(files, input_list, product_id,
                                                          d_type)
            
                    ## Save the file ##
                    ## try:
                    
                    # file_out = path_out+file_name
                    # harp.export_product(average_product, file_out+'.nc')
                    
                    print('DONE!!') # {product_id} for {st_nm}:
                    
                    ## except: # harp.Error
                    ##     print(f'{st_nm}: {color_code}{error_mess}')
                        ## pass
    
    else:
        
        if product_id == 'TO3_NRTI':
            
            main_inpath = '/mnt/lapaero_b/groups/lap_aero/PRIP/*/'
            data_path = 'TO3/'
            
            ## Make the output folder ##
            # path_out = tools.make_folder(main_outpath, data_path)
        
        else:
            
            inpath = '/mnt/lapsat/groups/lapsat/'
            
            if product_id == 'CLOUD_OFFL':
                data_path = 'TROPOMI_Cloud/'
            
            elif product_id == 'CH4_OFFL':
                data_path = 'TROPOMI_CH4/'
            
            else:
                data_path = 'TROPOMI_H2O/'
            
            main_inpath = inpath+data_path
            
            ## Make the output folder ##
            # path_out = tools.make_folder(main_outpath, product_id.split('_')[0]+'/')
        
        ## Get the year range from the folder names ##
        year_folders = glob.glob(main_inpath+f'{yr}/'); year_folders.sort()
        year_range = tools.year_range(year_folders, '/')
        
        ## Get the files ##
        if product_id in ['CH4_OFFL', 'TCWV_PAL']:
            files = glob.glob(main_inpath+f'{yr}/*/*')
        else:
            files = glob.glob(main_inpath+f'{yr}/*')
        files.sort()
        
        # files = files[:10] # For testing!!
        
        for st_nm in stations:
            
            file_name = '_'.join(['S5P', product_id, st_nm, 'Overpass',
                                  year_range])
            
            st_lat, st_lon = coordinates_dict[st_nm]
            
            if product_id in ['TO3_NRTI', 'CH4_OFFL', 'TCWV_PAL']:
                input_list = [st_lat, st_lon, point_val, qa_val]
            
            else:
                input_list = [st_lat, st_lon, point_val, qa_val, 'CRB', 'UVVIS']
            
            print(f'{product_id} for {st_nm}: Running...')
            
            ## Initialize the average Harp product, just in case ##
            average_product = 0
            
            ## Get the average/overpass data ##
            average_product = tools.s5p_avrg_ovps(files, input_list, product_id,
                                                  d_type)
    
            ## Save the file ##
            ## try:
            
            # file_out = path_out+file_name
            # harp.export_product(average_product, file_out+'.nc')
            
            print('DONE!!') # {product_id} for {st_nm}:
            
            ## except: # harp.Error
            ##     print(f'{st_nm}: {color_code}{error_mess}')
                ## pass

