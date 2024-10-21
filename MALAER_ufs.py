#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:49:56 2024

@author: pfountou
"""

'''
This file has several utility functions (ufs), which are used in the MALAER
algorithm.
'''

import os
import sys
import glob
import datetime as dtt # from datetime import datetime
import numpy as np
import sympy as sp
import xarray as xr
import pandas as pd
import netCDF4 as nc # from netCDF4 import Dataset
import harp
import coda
import functools
import math
import pytz


def date_to_datetime(date_arr, date_format):
    
    '''
    This function converts raw dates into datetime format.
    
    INPUTS:
    date_arr: array of date values (int/str)
    date_format: the format of the date values (str, e.g. '%Y%m%d%H%M',
    or %Y/%m/%d %H:%M, etc.)
    
    OUTPUTS:
    dt_date_arr: array of dates (datetime)
    
    NOTES:
    Infos about datetime format can be found in:
    https://www.w3schools.com/python/python_datetime.asp
    '''
    
    dt_date_arr = [dtt.datetime.strptime(str(x), date_format) for x in date_arr]
    dt_date_arr = np.array(dt_date_arr) # Turn the list into an array
    
    return dt_date_arr # An array of dates in datetime format

def datetime_to_doy(dt_date_arr):
    
    '''
    This function converts an array/list of datetime values into doy (day of year).
    
    INPUTS:
    dt_date_arr: array/list of dates (datetime)
    
    OUTPUTS:
    doy: array of the day of year (int)
    
    NOTES:
    More infos in: https://www.w3schools.com/python/python_datetime.asp
    '''
    
    doys = [int(x.strftime('%j')) for x in dt_date_arr]
    doys = np.array(doys) # Turn the list into an array
    
    return doys

def datetime_to_str(dt_date_arr, date_format):
    
    '''
    This function converts an array/list of datetime values into string.
    
    INPUTS:
    dt_date_arr: array/list of dates (datetime)
    date_format: the format of the datetime values (str, e.g. '%Y%m%d%H%M',
    or %Y/%m/%d %H:%M:%S, etc.)
    
    OUTPUTS:
    date_str: array/list of the dates (str)
    '''
    
    date_str = [x.strftime(date_format) for x in dt_date_arr]
    date_str = np.array(date_str)
    
    return date_str

def datetime_round(datetime_arr, round_par):
    
    '''
    This function rounds a datetime object in a specific time parameter (hour,
    day, minutes, etc.)
    
    INPUTS:
    datetime_arr: array/list of the datetime objects (datetime)
    round_par: the time parameter in which we want to round the datetime
    objects (str, 'H', 'T', 'D', etc.)
    
    OUTPUTS:
    datetime_round: the datetime array/list with the round values
    
    NOTES:
    More info about rounding a datetime object can be found in
    https://www.statology.org/how-to-round-dates-to-the-nearest-day-hour-or-minute-in-python/
    '''
    
    datetime_round = []
    for dtv in datetime_arr:
        
        dtv = pd.to_datetime(dtv); dtv = dtv.round(round_par)
        datetime_round.append(dtv)
        
    # datetime_round = np.array(datetime_round)
    
    return datetime_round

def datetime_range(tmval, trng, changepar, strpar):
    
    '''
    This function gives a specific +/- date/time values, from a certain
    date/time point value.
    
    INPUTS:
    tmval: the certain date/time point value (datetime)
    trng: the date/time range we want (int, e.g. +/- 10 minutes, or +/- 2
    hours, etc)
    changepar: the time parameter, in which we want the range (str, valid
    values: 'year', 'month', 'day', 'hour', 'minute', 'second')
    strpar: define whether we want the dt1 and dt2 values as str, or not (str,
    valid values: 'y', or 'n')
    
    OUTPUTS:
    dt1, dt2: the -/+ time range values, respectively, from the certain
    date/time point value (datetime)
    
    NOTES:
    If we want a list/array of date/time range values, e.g. start from a
    specific time point and add (or subtract) 10 minutes, we can do the
    following:
    
    dt1 = [tmval-dtt.timedelta(minutes = x) for x in range(trng, a*trng, trng)]
    dt2 = [tmval+dtt.timedelta(minutes = x) for x in range(trng, a*trng, trng)]
    
    where a defines how many times we will add the trng parameter.
    '''

    if changepar == 'year':
        
        dt1 = dtt.datetime(tmval.year-trng, tmval.month, tmval.day,
                           tmval.hour, tmval.minute, tmval.second)
        dt2 = dtt.datetime(tmval.year+trng, tmval.month, tmval.day,
                           tmval.hour, tmval.minute, tmval.second)

    elif changepar == 'month':
        
        extra_month = dtt.timedelta(days = 31*trng)
        dt1 = tmval-extra_month
        dt2 = tmval+extra_month

    elif changepar == 'day':
        
        dt1 = tmval-dtt.timedelta(days = trng)
        dt2 = tmval+dtt.timedelta(days = trng)

    elif changepar == 'hour':
        
        dt1 = tmval-dtt.timedelta(hours = trng)
        dt2 = tmval+dtt.timedelta(hours = trng)

    elif changepar == 'minute':
        
        dt1 = tmval-dtt.timedelta(minutes = trng)
        dt2 = tmval+dtt.timedelta(minutes = trng)

    else:
        
        dt1 = tmval-dtt.timedelta(seconds = trng)
        dt2 = tmval+dtt.timedelta(seconds = trng)
    
    ## Convert the datetime values into str ##
    if strpar == 'y':
        date_format = '%Y-%m-%d %H:%M:%S'
        dt1 = dt1.strftime(date_format); dt2 = dt2.strftime(date_format)
    
    return dt1, dt2

def timezone_conversion(dtdate_arr, oldtz, newtz):
    
    '''
    This function converts an array/list of datetime values from one timezone
    to another. The different timezones can be found in
    https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568
    
    INPUTS:
    dtdate_arr: array/list of datetime values
    oldtz: the timezone in which our datetime objects already are (str)
    newtz: the timezone in which we want to convert our datetime objects (str)
    
    OUTPUTS:
    utc_dt: array of datetime values in UTC
    
    NOTES:
    The following method, converts a datetime object in UTC, as well, but it
    takes as local timezone the timezone of the PC in which we run the script:
    utc_dt = [x.astimezone(dtt.UTC) for x in dtdate_arr] # or .astimezone(dtt.timezone.utc)
    '''
    
    tz = pytz.timezone(oldtz); tz2 = pytz.timezone(newtz); utc_dt = []
    for d in dtdate_arr:
        
        ## Define the timezone in which the datetime objects already are ##
        d = tz.localize(d)
        
        ## Change the timezone of the datetime objects to the new one ##
        utc_dt.append(d.astimezone(tz2))
    
    utc_dt = np.array(utc_dt)
    
    return utc_dt

def utc_sec_to_datetime(sec_arr, timestamp_val):
    
    '''
    This function converts the UTC time (in seconds) into full
    date in datetime format.
    
    INPUTS:
    sec_arr: array of time values in seconds (float)
    timestamp_val: the seconds between 01/01/1970 (Python start date)
    and the start date of the respective data (int)
    
    OUTPUTS:
    dt_arr: array of time values (datetime)
    
    NOTES:
    The timestamp_val is the seconds between 01/01/1970 (Python start date)
    and the start date of the respective data. The formula is the following:
    1 yr in seconds*(data initial yr - python initial yr) +/- some days in seconds.
    For example, for S5P/TROPOMI the value is:
    timestamp_val = 31536000*(2010 - 1970) + 864000 = 1262304000
    '''
    
    dt_arr = [dtt.datetime.utcfromtimestamp(x+timestamp_val) for x in sec_arr]
    dt_arr = np.array(dt_arr)
    
    return dt_arr

def multi_col_df(data_list, index_arr, main_cols, secondary_cols):
    
    '''
    This function gives a multi-column DataFrame (based on the master thesis
    analysis).
    
    INPUTS:
    data_list: list which contains the variables/data we want to insert to the DataFrame
    index_arr: array/list of the indexes of the DataFrame (int, float, datetime, etc.)
    main_cols: array/list of the main columns of the DataFrame (str)
    secondary_cols: array/list of the secondary columns of the DataFrame (str)
    
    OUTPUTS:
    df: the multi-column DataFrame
    '''
    
    df = pd.DataFrame()
    for var in range(len(data_list)):
                
        if var == 0:
            
            # Define the DataFrame at the first iteration and make the multi-columns #
            df = pd.DataFrame(data_list[var], index = index_arr, columns = secondary_cols)
            df.columns = pd.MultiIndex.from_product([[main_cols[var]], df.columns])
        
        else:
            
            # Add the other varibles to the DataFrame as multi-column format #
            df = df.join(pd.DataFrame(data_list[var], index = df.index,
                                      columns = pd.MultiIndex.from_product([[main_cols[var]],
                                                                            secondary_cols])))
    
    return df

def L1_weighted_average(wl_arr, L1_arr, wl_val, dwl): # , wldist
    
    '''
    This function calculates the weighted avergae of the L1 (radiance and/or irradiance)
    satellite measurements, based on the wavelength values. The weights of the
    average are calculated based on a triangular function.
    
    INPUTS:
    wl_arr: the 2D array with the wavelength values (int/float)
    L1_arr: the 2D array with the L1 (radiance and/or irradiance) values (int/float)
    wl_val: the specific wavelength value we want (int/float)
    dwl: the wavelength resolution we want, +/- the specific wavelength wl_val
    (int/float)
    # wldist: the 'distance' between the wavelength value(s) and the wl_val
    # (int/float) --> NO USE!!!
    
    OUTPUTS:
    L1_mean: the weighted average values (array, int/float)
    '''
    
    ## Define the the weights for the triangular function ##
    weights_arr = [1, 3]
    
    ## Make the wl +/- dwl array/list ##
    wls = np.array([wl_val-dwl, wl_val, wl_val+dwl])
    
    L1_mean = []
    for i in range(len(wl_arr)):
        
        ## Keep the L1 values for wl +/- dwl ##
        condition = np.logical_and(wl_arr[i] >= wl_val-dwl, wl_arr[i] <= wl_val+dwl)
        temp_wls = wl_arr[i][condition]; temp_l1 = L1_arr[i][np.where(condition)]
        
        ## Keep the weights for each L1 value ##
        w = [] # the weigths
        for v in temp_wls:
            
            idx = np.argmin(abs(wls - v))
            
            if idx == 1:
                w.append(weights_arr[1]) # for the center of the triangle
            else:
                w.append(weights_arr[0]) # for the edges of the triangle
            
        w = np.array(w)
        L1_mean.append(np.average(temp_l1, weights = w)) # the weighted average
    
    L1_mean = np.array(L1_mean)
    
    return L1_mean

def df_intersect(df1, df2):
    
    '''
    This function finds the common index values between two DataFrames, based
    on the second one, df2. It returns df1 with the new common index values.
    
    INPUTS:
    df1, df2: the two DataFrames
    
    OUTPUTS:
    df1_new: the DataFrame df1, with the new common index values
    '''
    
    common_idxs = df1.index.intersection(df2.index)
    df1_new = df1.loc[common_idxs]
    
    return df1_new

def make_2d_features(arr):
    
    '''
    This function makes a 2D array, which can be used as a features matrix for
    a Machine Learning model.
    
    INPUTS:
    arr: an array with values (int/float/str)
    
    OUTPUTS:
    newarr: the 2D features array
    '''
    
    if arr.ndim == 1:
        newarr = arr.reshape(-1, 1)
    else:
        newarr = arr
    
    return newarr

def read_AERONET_files(fls, vars_lst):
    
    '''
    This function reads AERONET files and extracts specific parameters
    (columns), as a DataFrame.
    
    INPUTS:
    fls: array/list of the AERONET files (str)
    vars_lst: list which contains the parameters (columns) we want to keep
    from the AERONET files (str)
    
    OUTPUTS:
    df: a DataFrame with the variables we chose
    '''
    
    ## Sort the files ##
    fls.sort()
    
    ## Loop for each file ##
    for f in range(len(fls)):
        
        ## Read the file ##
        fl = fls[f]
        temp_df = pd.read_csv(fl, skiprows = np.arange(7)) # skip the first 7 rows, if necessary
        temp_df.drop(temp_df.tail(1).index, inplace = True) # drop the last row, if necessary
    
        ## Keep specific variables (columns) from the file ##
        tdf = temp_df[vars_lst]
        
        ## Join the 'Date(dd:mm:yyyy)' and 'Time(hh:mm:ss)' columns ##
        tdf['Date'] = temp_df[['Date(dd:mm:yyyy)', 'Time(hh:mm:ss)']]\
            .agg(' '.join, axis = 1) # works only between strings
        
        ## Turn the -999 values into Nan and drop them ##
        tdf = tdf.replace(-999, np.nan); tdf = tdf.dropna()
        
        ## Join together the DataFrames for each file ##
        if f == 0:
            df = tdf
        else:
            df = pd.concat([df, tdf])
    
    ## Convert the dates in the 'Date' column into DateTime and exclude the seconds ##
    # df['Date'] = pd.to_datetime(df['Date'], format = '%d:%m:%Y %H:%M:%S')
    dt_date = date_to_datetime(df['Date'].values, '%d:%m:%Y %H:%M:%S')
    dt_date = [x.replace(second = 0) for x in dt_date]
    df['Date'] = dt_date
    
    ## Set the 'Date' column as Index ##
    df = df.set_index('Date')
    
    return df

def read_CAMS_files(fls, var, par_lst):
    
    '''
    This function reads CAMS (ADS or CDS) files and extracts specific
    parameters (columns). This function extracts only one meteorological
    parameter from the files.
    
    INPUTS:
    fls: array/list of the CAMS files (str)
    var: the name of the parameter that we want to extract, as it is written in
    the CAMS files (str)
    par_lst: list which contains specific parameters, depending on the fact that
    if the meteorological parameter we want is given in only one, or more than one
    pressure levels. If the meteorological parameter is given in only one pressure
    level then par_lst = ['single'], else par_lst = ['pressure', pressure_levels],
    where pressure_levels is the number of the pressure levels in which we will
    calculate the average value of the meteorological parameter
    
    OUTPUTS:
    the parameters that were extracted from the files, with the following
    order --> meteorological parameter, time, latitude, longitude
    
    NOTES:
    The latitude and longitude are the same for a specific area, regardless the
    time period. That's why we keep them only from the first file, given the fact
    that this function will be inside a for loop for several areas.
    '''
    
    ## Sort the files ##
    fls.sort()
    
    ## Loop for each file ##
    for f in range(len(fls)):
        
        ## Read the file ##
        fl = fls[f]
        ds = nc.Dataset(fl)
        
        ## Convert the time into DateTime format ##
        tm = ds.variables['time'][:]; tm = np.array(tm)
        tm = np.array([x*3600 for x in tm]) # convert hours to seconds
        tm = utc_sec_to_datetime(tm, 31536000*(1900 - 1970) - 1468800)

        ## Keep the meteorological variable ##
        meteo_par = ds.variables[var][:]; meteo_par = np.array(meteo_par)
        
        ## Keep the latitude and the longitude (only form the first file) ##
        if f == 0:
            lat = ds.variables['latitude'][:]; lat = np.array(lat)
            lon = ds.variables['longitude'][:]; lon = np.array(lon)
        
        ## Join together the data for each file ##
        if 'single' in par_lst:
            
            if f == 0:
                cams_dttime = list(tm)
                cams_data = meteo_par
            else:
                cams_dttime.extend(tm)
                cams_data = np.row_stack([cams_data, meteo_par])
        
        else:
            
            prlvls = par_lst[1]
            
            ## Convert the CAMS data from 4D into 3D ##
            td = []
            for l in range(len(meteo_par)):
                td.append(meteo_par[l, :prlvls].mean(axis = 0))
            td = np.array(td)
                        
            if f == 0:
                cams_dttime = list(tm)
                cams_data = td
            else:
                cams_dttime.extend(tm)
                cams_data = np.row_stack([cams_data, td])
    
    return cams_data, cams_dttime, lat, lon

def read_EEA_files(fls):
    
    '''
    This function reads European Environment Agency (EEA) files and
    extracts specific parameters (columns), as a DataFrame.
    
    INPUTS:
    fls: array/list of the EEA files (str)
    
    OUTPUTS:
    df: a DataFrame with the variables we chose
    
    NOTES:
    Some commands might need change, depending the pollutant/parameter we have.
    '''
    
    ## Sort the files ##
    fls.sort()
    
    ## Keep specific columns from the files ##
    df_cols = ['DatetimeBegin', 'Concentration', 'Validity', 'Verification']
    
    ## Loop for each file ##
    for i in range(len(fls)):
        
        ## Read the file ##
        tdf = pd.read_csv(fls[i], delimiter = ',') # hourly concentrations (in μg/m3 for PM2.5)
        tdf = tdf[df_cols]
        
        ## Convert the date/time into Datetime ##
        raw_date = tdf['DatetimeBegin'].values
        dtdate = date_to_datetime(raw_date, '%Y-%m-%d %H:%M:%S +01:00')
        
        ## Convert the date/time from CET (UTC+1) to UTC ##
        utc_dtdate = timezone_conversion(dtdate, 'CET', 'UTC')
        utc_dtdate = date_to_datetime(utc_dtdate, '%Y-%m-%d %H:%M:%S+00:00')
        
        ## Convert the raw dates ('DatetimeBegin' column) into DateTime ##
        # tdf['DatetimeBegin'] = pd.to_datetime(tdf['DatetimeBegin'],
        #                                      format = '%Y-%m-%d %H:%M:%S +01:00')
        tdf['DatetimeBegin'] = utc_dtdate
        
        ## Rename the 'DatetimeBegin' column ##
        tdf.rename(columns = {'DatetimeBegin':'Date'}, inplace = True)
        # , 'Concentration':'Concentration (μg/m3)' # NO NEED!!
        
        ## Sort the data by the 'Date' column ##
        tdf = tdf.sort_values(by = ['Date'])

        ## Filter the data ##
        tdf = tdf[(tdf['Validity'] == 1) & (tdf['Verification'] == 1)]
        tdf = tdf.dropna()
        
        ## Join together the DataFrames for each file ##
        if i == 0:
            df = tdf
        else:
            df = pd.concat([df, tdf])
    
    ## Keep only the 'Concentration' column and set the 'Date' column as Index ##
    df = pd.DataFrame(df['Concentration'].values, df['Date'].values,
                      columns = ['Concentration (μg/m3)'])
    # df = df.set_index('Date'); df = df['Concentration']
    
    return df

def make_CAMS_df(atm_par, tm_par):
    
    '''
    This function converts a CAMS atmospheric parameter from an array to a
    DataFrame.
    
    INPUTS:
    atm_par: the CAMS atmospheric parameter (array, int/float)
    tm_par: time array (datetime)
    
    OUTPUTS:
    df: the DataFrame
    '''
    
    if atm_par.ndim == 3:
        
        new_meteo_par = [] # np.nan*np.zeros(len(meteo_par))
        for i in range(len(atm_par)):
            
            data = atm_par[i]; data = data.reshape(-1, 1)
            data = data.T; data = data[0]; new_meteo_par.append(data)
        
        new_meteo_par = np.array(new_meteo_par)
        
        dfcols = [f'Pixel {i}' for i in range(1, new_meteo_par.shape[1]+1)]
        df = pd.DataFrame(new_meteo_par, tm_par, dfcols)
    
    else: # 4 dimensions
        
        new_meteo_par = []
        for i in range(atm_par.shape[1]):
            
            data = atm_par[:, i]; data2 = []
            for j in range(data.shape[0]):
                
                temp = data[j].reshape(-1, 1); temp = temp.T; temp = temp[0]
                data2.append(temp)
            
            data2 = np.array(data2)
            new_meteo_par.append(data2)
        
        dfcols = [f'Pressure L{i}' for i in range(1, atm_par.shape[1]+1)]
        dfcols2 = [f'Pixel {i}' for i in range(1, atm_par.shape[2]*atm_par.shape[3]+1)]
        df = multi_col_df(new_meteo_par, tm_par, dfcols, dfcols2)
    
    return df

