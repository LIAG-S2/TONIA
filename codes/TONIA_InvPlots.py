#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:09:29 2023

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
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


# %% import data

farmName = 'Trebbin Wertheim'
# farmName = 'Trebbin Aussenschlag'
# farmName = 'Beerfelde Banane'
# farmName = 'Beerfelde Fettke'
# farmName = 'Brodowin Hoben'
# farmName = 'Brodowin Seeschlag'
# farmName = 'Boossen_1211'
# farmName = 'Boossen_1601'

refPoints = f'{farmName} farm_refPoints.csv'
RefPoints = pd.read_csv(refPoints, delimiter=";")
refName = RefPoints['Name'].astype(int) # refPoint names
meanNearestPoints = np.loadtxt(f'{farmName}-meanNearestPoints.csv', delimiter=';')
data = np.loadtxt(f'{farmName} farm_data.csv', delimiter=';')
nearestArray = pd.read_csv(f'{farmName}-nearestArray.csv', delimiter=',') # data of individual closest points
Eutm, Nutm, H, Gamma = data[:,0] , data[:,1], data[:,2], data[:,8]
Rho1, Rho2, Rho3, Rho4, Rho6 = data[:, 3],data[:, 4],data[:, 5],data[:, 6],data[:, 7]

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

# %%  plots for Nearest points to Reference Points
with PdfPages(f'{farmName} Inversion_Result_ Mean&Indiv.pdf') as pdf:
# Inversion results for 'Mean' of closest points
    chi2VecIndiv = np.loadtxt(f'{farmName} _chi2_indiv.csv', delimiter=';')
    chi2VecMean = np.loadtxt(f'{farmName} _chi2_mean.csv', delimiter=';')
    for j, Data in enumerate(dataNearestMean):
        # plot individual data
        array = data
        dist = np.sqrt((array[:,0]-Data[1])**2+(array[:,1]-Data[2])**2)
        
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index-5, dist_index+5)
        # newamVec = np.tile(amVec, (len(data[nearestArray][:, 3:8]),1))
        Rho = np.array(Data[3:8])
        mydata = data[nearestArray][:, 3:8]


        Inv_indiv = np.loadtxt(f'{farmName} _InvResultIndividualPoints_{j}th point.csv', delimiter=';')
        Inv_mean = np.loadtxt(f'{farmName} _InvResultMeanPoints.csv', delimiter=';')
        InvMeanResponse = np.loadtxt(f'{farmName} _invMeanResponse_{j}th point.csv', delimiter=';')
        
        inv_mean = Inversion(fop=fop) # passing the fwd operator in the inversion
        fig = plt.figure(figsize=(15.5, 8))
        # plt.gca().set_title(f'ref-point {Data[0]} - {farmName}')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        spec = mpl.gridspec.GridSpec(ncols=3, nrows=2)
        ax0 = fig.add_subplot(spec[0,0])
        ax1 = fig.add_subplot(spec[0,1])
        ax2 = fig.add_subplot(spec[0,2])
        ax3 = fig.add_subplot(spec[1,0])
        ax4 = fig.add_subplot(spec[1,1])        
        ax5 = fig.add_subplot(spec[1,2])
       
        ax0.bar(range(len(chi2VecIndiv)), chi2VecIndiv[j], width = 0.3,  label="chi2 for individual data")
        ax0.axhline(chi2VecMean[j], linestyle='--', c='r', label="chi2 for mean data")
        ax0.grid(True)
        ax0.set_xlim([0, np.max(len(chi2VecIndiv))])
        ax0.legend(fontsize=8, loc=2)
        ax0.set_xlabel('individual data')
        ax0.set_ylabel('$\u03C7^2$')
        ax0.set_title('chi-square')
    
        ax1.semilogx(Rho, amVec, "+", markersize=9, mew=2, label="Mean Data")
        ax1.semilogx(InvMeanResponse, amVec, "x", mew=2, markersize=8, label="Response")
        ax1.invert_yaxis()
        ax1.set_xlim([5, 3000])
        ax1.grid(True) 
        ax1.semilogx(data[nearestArray][0, 3:8], amVec, ".", markersize=2, label="Individual Data")
        for i in range(1, mydata.shape[0]):
            ax1.semilogx(mydata[i, :], amVec, ".", markersize=2)
        ax1.legend(fontsize=8, loc=2)
        # ax1.set_title(f' Reference Point {Data[0]:.0f} - {farmName}',  loc='center', fontsize= 20)
        ax1.set_ylabel('spacing')
        ax1.set_title(f'Rhoa , {farmName} farm')
    
        
        drawModel1D(ax2, thickness=thk, values=Inv_mean[j], plot='semilogx', color='g', zorder=20, label="mean")
        ax2.set_xlim([5, 3000])
        ax2.legend(fontsize=8, loc=2)
        ax2.set_title('Model')
        for inv in Inv_indiv:        
            drawModel1D(ax2, thickness=thk, values=inv, plot='semilogx', color='lightgray', linewidth=1)
 
        ax3.plot(Eutm[nearestArray], Nutm[nearestArray], "x", markersize=8, mew=2, label='Nearest Points')

        RefP = RefPoints.sort_values('Name')    
        ax3.plot(RefPoints['E'][j], RefPoints['N'][j], "o", markersize=8, mew=2, label='Reference Point')
        ax3.axis("equal")
        ax3.set_title(f'Reference point {Data[0]:.0f} and its {len(nearestArray)} nearest points')
        ax3.legend(prop={'size': 8})
        ax3.set_xlabel('easting')
        ax3.set_ylabel('northing')
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.xaxis.set_tick_params(rotation=30)
        ax3.grid()

        matrixRho = np.vstack([Rho1[nearestArray],Rho2[nearestArray],Rho3[nearestArray],Rho4[nearestArray],Rho6[nearestArray]])
        norm=colors.LogNorm(vmin=10, vmax=3000)
        mat=ax4.matshow(matrixRho, cmap='Spectral_r', norm=norm)
        ax4.axis("equal")
        ax4.set_title('Rhoa')
        ax4.set_xlabel('nearest data to the reference point')
        # ax4.set_ylabel('spacing')
        # ax4.set_ylim([4,0])
        ax4.set_ylim( auto=True)
        # clb = plt.colorbar(mat, orientation='horizontal')
        # clb = ax4.colorbar()
        # clb.ax4.set_title('(ohm.m)',fontsize=10)
        fig.colorbar(mat, ax=ax4, orientation='horizontal')
        
        plt.xlabel('lines', fontsize=10)
        plt.ylabel('data', fontsize=10)
        plt.xticks(ticks=np.arange(len(matrixRho[0])))
        plt.yticks(ticks=np.arange(len(matrixRho)))
        # plt.xlim(0,9)
    
        pg.viewer.mpl.showStitchedModels(Inv_indiv.astype(int), ax=ax5, x=None, cMin=10, cMax=3000, cMap='Spectral_r', thk=thk, logScale=True, title='Model (Ohm.m)' , zMin=0, zMax=0, zLog=False)      
     
        plt.savefig(pdf, format='pdf')
        
# %% areal plots for all points
InvResultAll = np.loadtxt(f'{farmName} _InvResultAllPoints.csv', delimiter=';')
       
with PdfPages(f'{farmName}_ArealPlotAllPoints.pdf') as pdf:

    fig=plt.figure(figsize=(8, 6))
   
    for layer, column in enumerate(InvResultAll.T):
        norm=colors.LogNorm(vmin=10, vmax=1000, clip=False)
        ax = plt.scatter(Eutm, Nutm, s=0.5, c=column, cmap='Spectral_r', norm=norm)
        plt.axis("equal")
        plt.title(f'Inverted Resistivity for layer {layer+1}', fontsize=10)
        plt.suptitle(f'{farmName} farm', fontsize=14)
        plt.xlabel('Easting [m]')
        plt.ylabel('Northing [m]')
        plt.xticks(rotation=45, ha='right')    
        plt.yticks(rotation=45, ha='right')
        clb = plt.colorbar(ax)
        clb.ax.set_title('ohm.m',fontsize=8)    
        plt.savefig(pdf, format='pdf')  
        # ax.cla()
        fig.clf()
        
# %% areal plots downsampled Mean data (averaged)
InvResult_DownsampleMean = np.loadtxt(f'{farmName} _InvResultMeanDownSample.csv', delimiter=';')
      
with PdfPages(f'{farmName}_ArealPlot_MeanDownSampled.pdf') as pdf:
    data_dsMean = np.copy(data)
    meanN = 5 # every 'meanN' data are averaged
    while len(data_dsMean) % meanN != 0:
        data_dsMean = data_dsMean[:-1]    
    dataDivisibleby5 = np.mean(data_dsMean.reshape(((int(len(data_dsMean)/meanN)), meanN, 9)), axis=1)
    fig=plt.figure(figsize=(8, 6))
   
    for layer, column in enumerate(InvResult_DownsampleMean.T):
        norm=colors.LogNorm(vmin=10, vmax=1000, clip=False)
        ax = plt.scatter(dataDivisibleby5[:,0], dataDivisibleby5[:,1], s=0.5, c=column, cmap='Spectral_r', norm=norm)
        plt.axis("equal")
        plt.title(f'Mean Downsampled Inverted Resistivity for layer {layer+1}', fontsize=10)
        plt.suptitle(f'{farmName} farm', fontsize=14)
        plt.xlabel('Easting [m]')
        plt.ylabel('Northing [m]')
        plt.xticks(rotation=45, ha='right')    
        plt.yticks(rotation=45, ha='right')
        clb = plt.colorbar(ax)
        clb.ax.set_title('ohm.m',fontsize=8)    
        plt.savefig(pdf, format='pdf')  
        # ax.cla()
        fig.clf()
        
# %% areal plots for downsampled data
InvResult_Downsample = np.loadtxt(f'{farmName} _InvResultDownSample.csv', delimiter=';')
       
with PdfPages(f'{farmName}_ArealPlot_DownSampled.pdf') as pdf:

    fig=plt.figure(figsize=(8, 6))
   
    for layer, column in enumerate(InvResult_Downsample.T):
        norm=colors.LogNorm(vmin=10, vmax=1000, clip=False)
        ax = plt.scatter(Eutm[::4], Nutm[::4], s=0.5, c=column, cmap='Spectral_r', norm=norm)
        plt.axis("equal")
        plt.title(f'Downsampled Inverted Resistivity for layer {layer+1}', fontsize=10)
        plt.suptitle(f'{farmName} farm', fontsize=14)
        plt.xlabel('Easting [m]')
        plt.ylabel('Northing [m]')
        plt.xticks(rotation=45, ha='right')    
        plt.yticks(rotation=45, ha='right')
        clb = plt.colorbar(ax)
        clb.ax.set_title('ohm.m',fontsize=8)    
        plt.savefig(pdf, format='pdf')  
        # ax.cla()
        fig.clf()
    
