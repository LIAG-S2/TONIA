#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:05:11 2023

@author: sadegh
"""

#%% import liberaries
import numpy as np
import utm
import geopandas as gpd
import fiona
import pandas as pd
import pygimli as pg
from pygimli.frameworks import Modelling, Inversion

#%% Reading data
filepath = 'TRB_Wertheim_Geophilus_roh_221125.csv'
farmName = 'Trebbin Wertheim'

# filepath = 'Trebbin Aussenschlag.csv'
# farmName = 'Trebbin Aussenschlag'

# filepath = 'BFD_Banane_Geophilus_221118_roh.csv'
# farmName = 'Beerfelde Banane'

# filepath = 'BFD_Fettke_Geophilus_roh.csv'
# farmName = 'Beerfelde Fettke'

# filepath = 'BDW_Hoben_Geophilus_221020_roh.csv'
# farmName = 'Brodowin Hoben'

# filepath = 'BDW_Seeschlag_Geophilus_roh_221020 - Copy.csv'
# farmName = 'Brodowin Seeschlag'

# filepath = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName = 'Boossen_1211'

# filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName = 'Boossen_1601'

# EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Rho6, Gamma   \
#     = np.genfromtxt(filepath, skip_header=1, delimiter=';', unpack=True)
EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Rho6, Gamma, BFI, IfdNr, Date, time    \
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

# %% import data

refPoints = pd.read_csv(f'{farmName} farm_refPoints.csv', delimiter=';')
refName = refPoints['Name'].astype(int) # refPoint names
meanNearestPoints = np.loadtxt(f'{farmName}-meanNearestPoints.csv', delimiter=';')
data = np.loadtxt(f'{farmName} farm_data.csv', delimiter=';')
nearestArray = pd.read_csv(f'{farmName}-nearestArray.csv', delimiter=',') # data of individual closest points

# %% Forward Operator and response function
"""VES inversion."""

class VESRhoModelling(Modelling):
    """Vertical electrical sounding (VES) modelling with fixed layers."""

    def __init__(self, thk, **kwargs):
        super().__init__()
        self.fwd = pg.core.DC1dRhoModelling(thk, **kwargs) # kwargs: am, bm, an, bn
        
        mesh = pg.meshtools.createMesh1D(len(thk)+1)
        # self.clearRegionProperties()
        self.setMesh(mesh)

    def response(self, par):
        """Forward response."""
        return self.fwd.response(par)
    
# %% ABMN, res, thk 
# data space
# amVec = np.arange(1, 6) * 0.5  # Geophilius 1.0 (2013)
# amVec = np.arange(1, 7) * 0.5  # Geophilius 2.0 (2017)
amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
b = 1.0
bmVec = np.sqrt(amVec**2+b**2)

# model space
thk = np.ones(15) * 0.1
nLayer = len(thk) + 1

dataNearestMean = np.column_stack((np.array(refName, dtype = 'int'), meanNearestPoints))

# %% Initialize the DC Forward Modelling Operator
fop = VESRhoModelling(thk, am=amVec, an=bmVec, bm=bmVec, bn=amVec)

#  Error Model
error = np.ones_like(amVec) * 0.03 # used in inversion

# %% Inversion (For Nearest Points to Reference-Point)
refpoints = f'{farmName} farm_refPoints.csv'
RefPoints = pd.read_csv(refpoints, delimiter=";")
data = np.loadtxt(f'{farmName} farm_data.csv', delimiter=';')
Eutm, Nutm, H, Gamma = data[:,0] , data[:,1], data[:,2], data[:,8]
Rho1, Rho2, Rho3, Rho4, Rho6 = data[:, 3],data[:, 4],data[:, 5],data[:, 6],data[:, 7]
StmodelsMean = []  
chi2Vec_mean = []
chi2Vec_indiv = []
chi2_indiv = []

# with PdfPages(f'{farmName} Inversion_Result_ Mean&Indiv.pdf') as pdf:
# Inversion results for 'Mean' of closest points
for j, Data in enumerate(dataNearestMean):
    array = data
    dist = np.sqrt((array[:,0]-Data[1])**2+(array[:,1]-Data[2])**2)
    dist_index = np.argmin(dist)
    nearestArray = np.arange(dist_index-5, dist_index+5)
    newamVec = np.tile(amVec, (len(data[nearestArray][:, 3:8]),1)) 
    mydata = data[nearestArray][:, 3:8]
    chi2Vec_indiv = []
    Stmodels = []

    # inversion of individual nearest points
    for i, indivData in  enumerate(mydata):
        inv_indiv = Inversion(fop=fop) # passing the fwd operator in the inversion
        inv_indiv.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
        modelInd = inv_indiv.run(indivData, error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization
        Stmodels.append(modelInd)
        np.savetxt(f'{farmName} _InvResultIndividualPoints_{j}th point.csv', Stmodels, delimiter=';', fmt='%s')
        chi2 = inv_indiv.inv.chi2()
        chi2Vec_indiv.append(chi2)
    chi2_indiv.append(chi2Vec_indiv)
    np.savetxt(f'{farmName} _chi2_indiv.csv', chi2_indiv, delimiter=';', fmt='%s')


    # inversion of mean of the nearest points
    Rho = np.array(Data[3:8]) 
    inv_mean = Inversion(fop=fop) # passing the fwd operator in the inversion
    inv_mean.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
    modelMean = inv_mean.run(Rho, error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization
    invMeanResponse = inv_mean.response.array()
    StmodelsMean.append(modelMean)
    np.savetxt(f'{farmName} _InvResultMeanPoints.csv', StmodelsMean, delimiter=';', fmt='%s')
    np.savetxt(f'{farmName} _invMeanResponse_{j}th point.csv', invMeanResponse, delimiter=';', fmt='%s')
    chi2_m = inv_mean.inv.chi2()
    chi2Vec_mean.append(chi2_m)
    np.savetxt(f'{farmName} _chi2_mean.csv', chi2Vec_mean, delimiter=';', fmt='%s')

# %% Inversion for all points
# 1st doing the inversion then viualize areal plots
invAll = Inversion(fop=fop) # passing the fwd operator in the inversion
invAll.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
RES_All = np.zeros((len(data), 16))
# for i, point in enumerate(data[::6]):
#     Rho = point[3:8] 
#     modelMean = inv.run(Rho, error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization  
#     RES[i] = modelMean
#     Resmodel.append(modelMean)

for i, point in enumerate(data):
    RES_All[i] = invAll.run(point[3:8], error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization  

np.savetxt(f'{farmName} _InvResultAllPoints.csv', RES_All, delimiter=';', fmt='%s')

# %% Inversion for downsampled Mean data (averaged)
# 1st doing the inversion then viualize areal plots
invDownsampleMean = Inversion(fop=fop) # passing the fwd operator in the inversion
invDownsampleMean.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint

data_dsMean = np.copy(data)
meanN = 5 # every 'meanN' data are averaged
while len(data_dsMean) % meanN != 0:
    data_dsMean = data_dsMean[:-1]    
dataDivisibleby5 = np.mean(data_dsMean.reshape(((int(len(data_dsMean)/meanN)), meanN, 9)), axis=1)

RES = np.zeros((len(dataDivisibleby5), nLayer))

for i, point in enumerate(dataDivisibleby5):
    RES[i] = invDownsampleMean.run(point[3:8], error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization  

np.savetxt(f'{farmName} _InvResultMeanDownSample.csv', RES, delimiter=';', fmt='%s')

# %% Inversion for downsampled data
# 1st doing the inversion then viualize areal plots
invDownsample = Inversion(fop=fop) # passing the fwd operator in the inversion
invDownsample.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
RES = np.zeros((len(data[::4]), nLayer))

for i, point in enumerate(data[::4]):
    RES[i] = invDownsample.run(point[3:8], error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization  

np.savetxt(f'{farmName} _InvResultDownSample.csv', RES, delimiter=';', fmt='%s')

# %%

