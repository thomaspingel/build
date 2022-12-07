# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 21:48:45 2022

@author: tpingel
"""
import blickfeld_scanner
from blickfeld_scanner.protocol.config import scan_pattern_pb2
import numpy as np
import pdal
import time
import datetime
import os
import smrf
#import neilpy

import matplotlib.pyplot as plt
from matplotlib import cm

import rasterio

import multiprocessing as mp
from joblib import Parallel, delayed

import requests
import json


#%%

webhook = 'https://hooks.slack.com/services/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'

subnet = '192.168.10.'
targets = [101,103,104,105,106,107,108,109,110,111,112]
#targets = [101,103,104,105,106,107,108,109,110,111]
bbox = ((-12., -20., -0.5),(45., 25., 3.25))
cmap_name = 'gray'

lidar_delay_sec = 10
image_delay_sec = 1

lidar_resolution = 0.001
image_resolution = 0.05

low_floor_height = .4
high_floor_height = .8


outdir = '/lidar_data/data/out/'

#%%

# Derivative metrics
mins,maxes = bbox[0],bbox[1]
max_indexed_z = int(maxes[2] / image_resolution)

# Define colormap for later use
this_cm = cm.get_cmap(cmap_name, max_indexed_z+1)
cmap = {}
for i in range(max_indexed_z+1):
    cmap[i] = np.round(255*np.array(this_cm(i))).astype(np.uint8)
cmap[255] = (255,255,255,0)

# Dtypes for lidar
dtypes=np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('intensity', '<u2'),('PointSourceId','<u2')])



#%% Prepare devices for reading
devices = []

for target in targets:
    ip = subnet + str(target)
    devices.append(blickfeld_scanner.scanner(ip))
    
point_filter = scan_pattern_pb2.ScanPattern().Filter()
point_filter.max_number_of_returns_per_point = 1 # Set max number of returns to 2. The default value is 1.
point_filter.delete_points_without_returns = True # Filter points with no returns. This reduces the dataset only to valid returns.

streams = []
for device in devices:
    streams.append(device.get_point_cloud_stream(point_filter=point_filter, as_numpy=True) )

#%%

blickfeld_scanner.scanner.sync(devices=devices, target_frame_rate=1)
time.sleep(10)
    
#%%
def pollit(stream):
    frame,data = stream.recv_frame_as_numpy()
    return frame, data

#%%

# This size should be calculated, not hard-coded!
high_mask = np.zeros((900,1140),dtype=bool)
high_mask[540:,:] = True
high_mask[400:600,555:650] = True




#%%
error_state = False
num_sequential_errors = 0

requests.post(webhook,json.dumps({'text':'CID lidar tracking now starting.'}))


while True:
    try:
        # Pull datetime, and create output directories if needed
        dt = datetime.datetime.now()
        tic = time.time()
        secs = dt.second
        datestring = dt.strftime('%Y%m%d')
        if not os.path.exists(outdir + datestring):
            os.makedirs(outdir + datestring)
            requests.post(webhook,json.dumps({'text':'Message from the CID: It is a new day!  Logging continues.'}))
        if not os.path.exists(outdir + datestring + '/laz'):
            os.makedirs(outdir + datestring + '/laz')
        if not os.path.exists(outdir + datestring + '/max'):
            os.makedirs(outdir + datestring + '/max')
        if not os.path.exists(outdir + datestring + '/intensity'):
            os.makedirs(outdir + datestring + '/intensity')
        
        X,Y,Z,I,N = [],[],[],[],[]
        #for idx,stream in enumerate(streams):
        #    frame, data = stream.recv_frame_as_numpy()
        #    x, y, z, i = (data['cartesian']['x'], data['cartesian']['y'], data['cartesian']['z'],data['intensity'])
        #    X.append(x)
        #    Y.append(y)
        #    Z.append(z)
        #    I.append(i)
        #    N.append(targets[idx]*np.ones(frame.total_number_of_points,dtype=int))
        results = Parallel(n_jobs=mp.cpu_count(), prefer="threads")(delayed(pollit)(s) for s in streams)
        for idx,result in enumerate(results):
            frame, data = result[0], result[1]
            x, y, z, i = (data['cartesian']['x'], data['cartesian']['y'], data['cartesian']['z'],data['intensity'])
            X.append(x)
            Y.append(y)
            Z.append(z)
            I.append(i)
            N.append(targets[idx]*np.ones(len(x),dtype=int))
            
        
        
            
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        I = np.concatenate(I)   
        N = np.concatenate(N)
        
        # Run a bounding box
        idx = (X>mins[0]) & (X<maxes[0]) & (Y>mins[1]) & (Y<maxes[1]) & (Z>mins[2]) & (Z<maxes[2])
        X = X[idx]
        Y = Y[idx]
        Z = Z[idx]     
        I = I[idx]
        N = N[idx]
        
        if np.mod(secs,lidar_delay_sec)==0:
            fn = outdir + dt.strftime('%Y%m%d') + '/laz/' +  dt.strftime('%Y%m%d-%H%M%S') + '.laz'
            pipeline = pdal.Writer.las(
                filename=fn,
                offset_x=0,
                offset_y=0,
                offset_z=0,
                scale_x=lidar_resolution,
                scale_y=lidar_resolution,
                scale_z=lidar_resolution,
            ).pipeline(np.array(list(zip(X,Y,Z,I,N)),dtype=dtypes))
            pipeline.execute()
        
        if np.mod(secs,image_delay_sec)==0:
            
            Z[Z<0] = 0
            
            x_edges = np.arange(mins[0],maxes[0] + image_resolution,image_resolution)
            y_edges = np.arange(maxes[1],mins[1] - image_resolution,-image_resolution)
            
            [MAX,R] = smrf.create_dem(X,Y,Z,bin_type='max',edges=(x_edges,y_edges))
        
            nan_mask = np.isnan(MAX)
            
            # height_mask
            low_heights = (MAX < low_floor_height) | ((MAX < high_floor_height) & (high_mask)) 
            nan_mask = (nan_mask) | (low_heights)
            
            MAX = np.round(max_indexed_z*(MAX / maxes[2])).astype(np.uint8)
            MAX[nan_mask] = 255
        
            [INTENSITY,R] = smrf.create_dem(X,Y,I,bin_type='max',edges=(x_edges,y_edges))
            INTENSITY = np.round(254*(INTENSITY / 10000)).astype(np.uint8)
            INTENSITY[nan_mask] = 255
            
            metadata = {'compress':'lzw','dtype': 'uint8', 'width': np.shape(MAX)[1], 
                        'height': np.shape(MAX)[0], 'count':1, 'crs': None, 
                        'transform': R, 'nodata':255}
            fn = outdir + dt.strftime('%Y%m%d') + '/max/' +  dt.strftime('%Y%m%d-%H%M%S') + '.max.tif'
            with rasterio.open(fn, 'w', **metadata) as dst:
                dst.write(MAX, 1)
                dst.write_colormap(1,cmap)
            fn = outdir + dt.strftime('%Y%m%d') + '/intensity/' +  dt.strftime('%Y%m%d-%H%M%S') + '.intensity.tif'
            with rasterio.open(fn, 'w', **metadata) as dst:
                dst.write(INTENSITY, 1)
                dst.write_colormap(1,cmap)
            num_sequential_errors = 0
    
    except:
        print('failed frame at ',dt.strftime('%Y%m%d-%H%M%S'))
        num_sequential_errors = num_sequential_errors + 1
        if num_sequential_errors == 10:
            requests.post(webhook,json.dumps({'text':'CID Laptop is in an error state!'}))

            
    
    toc = time.time()
    wait_time = 1 - (toc-tic)
    if wait_time < 0:
        wait_time = 0
    time.sleep(wait_time)




