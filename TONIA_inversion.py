#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:27:05 2023

@author: sadegh
"""
# %%
import numpy as np
import pygimli as pg
from pygimli.frameworks import Modelling, Inversion
from pygimli.viewer.mpl import drawModel1D
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %% import data

# farmName = 'Brodowin Hoben'

# farmName = 'Beerfelde Banane'

farmName = 'Trebbin Wertheim'

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
 
# %% Inversion

with PdfPages(f'{farmName} Inversion_Result_ Mean&Indiv.pdf') as pdf:
# Inversion results for 'Mean' of closest points
    for Data in dataNearestMean:
        # plot individual data
        array = data
        dist = np.sqrt((array[:,0]-Data[1])**2+(array[:,1]-Data[2])**2)
        
        # distToRef = 8
        # nearestArray = np.nonzero(dist<distToRef)[0] # return indices of the elements that are non-zero
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index-5, dist_index+5)
        newamVec = np.tile(amVec, (len(data[nearestArray][:, 3:8]),1))
        fig, ax = pg.plt.subplots(figsize=(15, 6), ncols=4)  # two-column figure

        mydata = data[nearestArray][:, 3:8]
        
        chi2Vec_indiv = []
        chi2Vec_mean = []
        Stmodels = []
        for indivData in mydata:
            inv_indiv = Inversion(fop=fop) # passing the fwd operator in the inversion
            inv_indiv.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
            modelInd = inv_indiv.run(indivData, error, lam=20, startModel=100, verbose=True) # stating model 100 ohm.m, lam: regularization
            Stmodels.append(modelInd)
            drawModel1D(ax[0], thickness=thk, values=modelInd, plot='semilogx', color='lightgray', linewidth=1)
            chi2 = inv_indiv.inv.chi2()
            chi2Vec_indiv.append(chi2)
            # np.savetxt(f'chi2_indiv_{farmName} ref {Data[0]}.csv', chi2Vec_indiv, fmt='%s', delimiter=',')
            
        Rho = np.array(Data[3:8]) 
        inv_mean = Inversion(fop=fop) # passing the fwd operator in the inversion
        inv_mean.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
        modelMean = inv_mean.run(Rho, error, lam=10, startModel=100, verbose=True) # stating model 100 ohm.m, lam: regularization
        # print(modelMean)
        chi2_m = inv_mean.inv.chi2()
        chi2Vec_mean.append(chi2_m)
        # np.savetxt(f'chi2_mean_{farmName} ref {Data[0]}.csv', chi2Vec_mean, fmt='%s', delimiter=',')
                # plot model (inverted and synthetic)                
                
        drawModel1D(ax[0], thickness=thk, values=modelMean, plot='semilogx', color='g', zorder=20, label="mean")
        ax[0].set_xlim([0, 3000])
        ax[0].legend(fontsize=8, loc=2)
        ax[0].set_title(f'Mean data inversion close to ref-point {Data[0]} - {farmName}',  loc='left')

        ax[1].semilogx(Rho, amVec, "+", markersize=9, mew=2, label="Mean Data")
        ax[1].semilogx(inv_mean.response.array(), amVec, "x", mew=2, markersize=8, label="response")
        ax[1].invert_yaxis()
        ax[1].set_xlim([10, 3000])
        ax[1].grid(True) 
        ax[1].semilogx(data[nearestArray][0, 3:8], amVec, ".", markersize=2, label="Individual Data")
        for i in range(1, mydata.shape[0]):
            ax[1].semilogx(mydata[i, :], amVec, ".", markersize=2)
        ax[1].legend(fontsize=8, loc=2)
        
        # ax[2].plot(chi2Vec_indiv, "o", markersize=2, mew=2, label="chi2_data")
        ax[2].bar(range(len(chi2Vec_indiv)), chi2Vec_indiv, width = 0.3,  label="chi2 for individual data")
        # ax[2].plot(chi2Vec_mean, "*", markersize=4, mew=2, label="chi2_mean data")
        ax[2].axhline(chi2Vec_mean, linestyle='--', c='r', label="chi2 for mean data")
        ax[2].grid(True)
        ax[2].set_xlim([0, np.max(len(chi2Vec_indiv))])
        ax[2].legend(fontsize=8, loc=2)

        pg.viewer.mpl.showStitchedModels(Stmodels, ax=ax[3], x=None, cMin=30, cMax=1000, cMap='Spectral_r', thk=thk, logScale=True, title='stichmodel (Ohm.m)' , zMin=0, zMax=0, zLog=False)      

        plt.savefig(pdf, format='pdf') 
