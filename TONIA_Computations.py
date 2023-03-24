#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:16:18 2023

@author: sadegh
"""

#%% import liberaries
import numpy as np
import utm
import geopandas as gpd
import fiona
import pandas as pd


#%% Reading data
filepath = 'TRB_Wertheim_Geophilus_roh_221125.csv'
farmName = 'Trebbin Wertheim'

EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Rho6, Gamma, BFI, IfdNr, Date, time  \
    = np.genfromtxt(filepath, skip_header=1, delimiter=';', unpack=True)
dataRaw = np.genfromtxt(filepath, names=True, delimiter=';')
Eutm, Nutm, zone, letter  = utm.from_latlon(NdecDeg, EdecDeg) # convert E&N from DecimalDegree to UTM
data = np.column_stack((Eutm, Nutm, H, Rho1, Rho2, Rho3, Rho4, Rho6, Gamma,))
np.savetxt(f'{farmName} farm_data.csv', data, delimiter=';', fmt='%s')

# import kml of reference points
kmlFile = "TRB_Wertheim_Referenzpunkte_gemessen_2022-12-01.kml"
fiona.drvsupport.supported_drivers['KML'] = 'rw'
gdf = gpd.read_file(kmlFile) #GeoDataFrame object (ref names & geometry,in decimaldegree)
refLat = np.array(gdf['geometry'].y)
refLon = np.array(gdf['geometry'].x)
refEutm, refNutm, refzone, refletter  = utm.from_latlon(refLat, refLon) # convert E&N from DecimalDegree to UTM
refName = np.array(gdf['Name'])
refPoints = np.column_stack([refName, refEutm, refNutm])# RefPointNames+geometry,in lat/long
header = ['Name', 'E', 'N']
refPointsTable = np.vstack((header, refPoints))
np.savetxt(f'{farmName} farm_refPoints.csv', refPointsTable, delimiter=';', fmt='%s')

# %% Find nearest dataPoints to refPoints
refName = np.array(gdf['Name'])
meanDataVal =[]
refpoints = f'{farmName} farm_refPoints.csv'
RefPoints = pd.read_csv(refpoints, delimiter=";")

for point in refPoints:
    array = data
    dist = np.sqrt((array[:,0]-point[1])**2+(array[:,1]-point[2])**2)
    # distToRef = 8
    # nearestArray = np.nonzero(dist<distToRef)[0] # return indices of the elements that are non-zero
    dist_index = np.argmin(dist)
    nearestArray = np.arange(dist_index-5, dist_index+5)
    # npoints = 5
    # nearestArayRow = np.argsort(np.abs(nearestArray - np.mean(nearestArray)))[:npoints]
    
    dataNearest = data[nearestArray]   # data of individual closest points
    
    MeanOfNearestPoints = np.column_stack((np.mean(Eutm[nearestArray]), np.mean(Nutm[nearestArray]), \
                              np.mean(Rho1[nearestArray]), np.mean(Rho2[nearestArray]), \
                              np.mean(Rho3[nearestArray]), np.mean(Rho4[nearestArray]), \
                              np.mean(Rho6[nearestArray]), 
                              np.mean(Gamma[nearestArray])))        
    meanDataVal.append(MeanOfNearestPoints)
    meanDataArray = np.array(meanDataVal)
    meanDataNearestVector = np.array((meanDataArray[:,0,:])) # mean data of closest points
np.savetxt(f'{farmName}-meanNearestPoints.csv', meanDataNearestVector, delimiter=';')
np.savetxt(f'{farmName}-nearestArray.csv', dataNearest, delimiter=',', fmt='%s')

# %% 