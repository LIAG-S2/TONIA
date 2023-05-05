#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:15:10 2023

@author: sadegh
"""
#%% import liberaries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import pygimli as pg
import pandas as pd
import skgstat as skg
import matplotlib.colors as colors

# %% Save data as a multipage pdf 

# import data
# filepath = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName = 'Trebbin Wertheim'

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

filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
farmName = 'Boossen_1601'

data = np.loadtxt(f'{farmName} farm_data.csv', delimiter=';')
Eutm, Nutm, H, Gamma = data[:,0] , data[:,1], data[:,2], data[:,8]
Rho1, Rho2, Rho3, Rho4, Rho6 = data[:, 3],data[:, 4],data[:, 5],data[:, 6],data[:, 7]
refPoints = pd.read_csv(f'{farmName} farm_refPoints.csv', delimiter=';')
refEutm, refNutm = refPoints['E'] , refPoints['N']
RhoData = data[:, 3:8]

with PdfPages(f'{farmName} mapped-data-multipage_pdf_.pdf') as pdf:
    # Orthophoto map
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Eutm, Nutm, ".", markersize=0.5)
    ax.set_aspect(1.0)
    ax.grid()
    plt.title('Digital-Orthophoto', fontsize=10)
    plt.suptitle(f'{farmName} farm', fontsize=14)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.yticks(rotation=45, ha='right')
    plt.xticks(rotation=45, ha='right')
    plt.axis("equal")
    pg.viewer.mpl.underlayBKGMap(ax, mode='DOP', utmzone=33, epsg=0, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3', usetls=False, origin=None) 
    plt.savefig(pdf, format='pdf') 
            
    # Elevation  map
    fig=plt.figure(figsize=(8, 6))      
    ax2 = plt.scatter(Eutm, Nutm, s=0.5, c=H, cmap='plasma', vmin=np.min(H), vmax=np.max(H))
    clb = plt.colorbar(ax2)
    clb.ax.set_title('[m]',fontsize=8)
    plt.axis("equal")
    plt.title('Elevation', fontsize=10)
    plt.suptitle(f'{farmName} farm', fontsize=14)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.yticks(rotation=45, ha='right')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(pdf, format='pdf') 
    
    # Gamma map
    fig=plt.figure(figsize=(8, 6))      
    ax2 = plt.scatter(Eutm, Nutm, s=0.5, c=Gamma, cmap='plasma', vmin=0.4, vmax=1.6)
    clb = plt.colorbar(ax2)
    clb.ax.set_title('[?]',fontsize=8)
    plt.axis("equal")
    plt.title('Gamma', fontsize=10)
    plt.suptitle(f'{farmName} farm', fontsize=14)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.xticks(rotation=45, ha='right')    
    plt.yticks(rotation=45, ha='right')
    plt.savefig(pdf, format='pdf')     
    
    # Rho maps
    for col in range(RhoData.shape[1]):
        Rho_i = col+1
        Rhoi =RhoData[:,col]
        norm = matplotlib.colors.LogNorm(vmin=10, vmax=1000, clip=False)
        fig=plt.figure(figsize=(8, 6))
        ax1 = plt.scatter(Eutm, Nutm, s=0.5, c=RhoData[:,col], cmap='Spectral_r', norm=norm)
        clb = fig.colorbar(ax1)
        clb.ax.set_title('(ohm m)',fontsize=8)
        plt.axis("equal")
        plt.title(f'Rho{Rho_i}', fontsize=10)
        plt.suptitle(f'{farmName} farm', fontsize=14)
        plt.xlabel('Easting [m]')
        plt.ylabel('Northing [m]')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        ax2 = plt.scatter(refEutm, refNutm, s=5.5, label="Reference points") #Ref Points
        refName = refPoints['Name'].astype(int) # refPoint names
        for i, txt in enumerate(refName):
            plt.annotate(txt, (refEutm[i], refNutm[i]), fontsize=7)
            plt.legend(loc="upper left")
        plt.savefig(pdf, format='pdf')   
    
    # histogram of data distribution (log)
    RhoData = np.column_stack((Rho1, Rho2, Rho3, Rho4, Rho6))
    num_bins = 100
    fig=plt.figure(figsize=(8, 6))
    c = 1  # initialize plot counter
    for cols in range(RhoData.shape[1]):
        Rho_i = cols+1
        Rhoi =RhoData[:,cols]
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, c, sharex=ax1, sharey=ax1)
        counts, bins = np.histogram(np.log10(RhoData[:,cols]), bins=num_bins)
        ax3 = plt.stairs(counts, bins)
        ax4 = plt.hist(bins[:-1], bins, weights=counts, alpha=0.7)
        plt.suptitle(f'Data Frequency Distributions - {farmName} farm', fontsize=15, y=0.98)
        plt.title(f'Rho{Rho_i}', fontsize=12)
        plt.xlabel('log Resistivity')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.25, hspace=0.4)
        plt.grid()
        c=c+1

    plt.savefig(pdf, format='pdf')       
# %% Find nearest dataPoints to refPoints
meanDataVal =[]
refpoints = f'{farmName} farm_refPoints.csv'
RefPoints = pd.read_csv(refpoints, delimiter=";")

# distToRef = 8
    
with PdfPages(f'{farmName} DataDistNearestPoints.pdf') as pdf:
    for i , point in RefPoints.iterrows():
        # print(point)
        array = data
        dist = np.sqrt((array[:,0]-point['E'])**2+(array[:,1]-point['N'])**2)

        # nearestArray = np.nonzero(dist<distToRef)[0] # return indices of the elements that are non-zero
        
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index-5, dist_index+5)
        dataNearest = data[nearestArray]   # data of individual closest points
        
        fig, ((ax1,ax2,ax3), (ax4,ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
        ax1.semilogy(Rho1[nearestArray], marker='.', label='Rho1')
        ax1.set_title('Rho1')
        ax1.grid()
        fig.suptitle(f'Data distribution of closer points to refrence point {point[0]} - {farmName}', fontsize=16)

        ax2.semilogy(Rho2[nearestArray], linestyle=None, marker='.', label='Rho2')
        ax2.set_title('Rho2')
        ax2.grid()
        
        ax3.plot(Eutm[nearestArray], Nutm[nearestArray], "x", markersize=8, mew=2, label='Nearest Points')
        ax3.plot(point[1], point[2], "o", markersize=8, mew=2, label='Reference Point')
        ax3.axis("equal")
        ax3.set_title('Points´ location')
        ax3.legend(prop={'size': 8})
        
        ax4.semilogy(Rho3[nearestArray], linestyle=None, marker='.', label='Rho3')
        ax4.set_title('Rho3')
        ax4.grid()
        
        ax5.semilogy(Rho4[nearestArray], linestyle=None, marker='.', label='Rho4')
        ax5.set_title('Rho4')
        ax5.grid()
                      
        ax6.semilogy(Rho6[nearestArray], linestyle=None, marker='.', label='Rho6')
        ax6.set_title('Rho6')
        ax6.grid()
            
        plt.savefig(pdf, format='pdf') 
# %%   
    
with PdfPages(f'{farmName} RhoaDistNearestPoints.pdf') as pdf:
    for i , point in RefPoints.iterrows():
        array = data
        dist = np.sqrt((array[:,0]-point['E'])**2+(array[:,1]-point['N'])**2)   
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index-5, dist_index+5)
        dataNearest = data[nearestArray]   # data of individual closest points
        
        fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(9.5, 10))
        fig.suptitle(f'Rhoa distribution of closer points to refrence point {point[0]} - {farmName}', fontsize=16)
        
        ax1.plot(Eutm[nearestArray], Nutm[nearestArray], "x", markersize=8, mew=2, label='Nearest Points')
        ax1.plot(point[1], point[2], "o", markersize=8, mew=2, label='Reference Point')
        ax1.axis("equal")
        ax1.set_title('Points´ location', fontweight='bold')
        ax1.legend(prop={'size': 8})
        
        matrixRho = np.vstack([Rho1[nearestArray],Rho2[nearestArray],Rho3[nearestArray],Rho4[nearestArray],Rho6[nearestArray]])
        norm=colors.LogNorm(vmin=30, vmax=1000)
        mat=ax2.matshow(matrixRho, cmap='Spectral_r', norm=norm)
        
        ax2.axis("equal")
        ax2.set_title('Rhoa' , fontweight='bold')
        clb = plt.colorbar(mat)
        # clb = fig.colorbar()
        clb.ax.set_title('(ohm.m)',fontsize=10)
        plt.xlabel('Points', fontsize=10)
        plt.ylabel('data', fontsize=10)
        plt.xticks(ticks=np.arange(len(matrixRho[0])))
        plt.yticks(ticks=np.arange(len(matrixRho)))
        plt.xlim(0,9)
               
        fig.tight_layout(pad=3.0)
        plt.savefig(pdf, format='pdf') 
# %%   

with PdfPages(f'{farmName} DataHistogramNearestPoints.pdf') as pdf:
    for i , point in RefPoints.iterrows():
        array = data
        dist = np.sqrt((array[:,0]-point['E'])**2+(array[:,1]-point['N'])**2)
        # nearestArray = np.nonzero(dist<distToRef)[0] # return indices of the elements that are non-zero
        dist_idx = np.argmin(dist)
        nearestArrays = np.arange(dist_idx-5, dist_idx+5)
        Data_NearestPoints = np.column_stack(((Eutm[nearestArrays]), (Nutm[nearestArrays]), \
                                  (Rho1[nearestArrays]), (Rho2[nearestArrays]), \
                                  (Rho3[nearestArrays]), (Rho4[nearestArrays]), \
                                  (Rho6[nearestArrays]), 
                                  (Gamma[nearestArrays]))) 
          
        Rho_Nearest = Data_NearestPoints[:, 2:7] # Rho of individual closest points
        num_bins = 25
        fig=plt.figure(figsize=(8, 6))
        fig.text(0.5, 0.04, 'log Resistivity', ha='center')
        fig.text(0.04, 0.5, '', va='center', rotation='vertical')
        c = 1  # initialize plot counter
        for cols in range(Rho_Nearest.shape[1]):
            ### plot histogram
            Rho_i = cols+1
            Rhoi =Rho_Nearest[:,cols]
            # print(f'mean Rho{Rho_i} = ', np.round(np.mean(Rhoi),2),  f' \ std Rho{Rho_i} = ', np.round(np.std(Rhoi),2))
            ax1 = plt.subplot(2, 3, 1)
            ax2 = plt.subplot(2, 3, c, sharex=ax1, sharey=ax1)
            counts, bins = np.histogram(np.log10(Rhoi), bins=num_bins)
            ax3 = plt.stairs(counts, bins)
            ax4 = plt.hist(bins[:-1], bins, weights=counts, alpha=0.7)
            plt.suptitle(f'Distribution of the Nearest Data to refrence point {point[0]}- {farmName} farm', fontsize=13, y=0.98)
            plt.title(f'Rho{Rho_i}', fontsize=12)
            # plt.xlabel('log Resistivity')
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.25, hspace=0.7)
            plt.grid()
            c=c+1
           
        plt.savefig(pdf, format='pdf') 
# %%
"""Experimental variograms."""
with PdfPages(f'{farmName}_variograms.pdf') as pdf:
    
    Rho = np.column_stack((Rho1, Rho2, Rho3, Rho4, Rho6))
    for rho in range(Rho.shape[1]):
        rho_i = rho+1
        rhoi =Rho[:,rho]    
        coords = np.column_stack((Eutm, Nutm))
        V = skg.Variogram(coords, np.log(rhoi),
                          n_lags=30, maxlag=500, model="exponential")
        print(V)
        fig = V.plot()
        
        plt.savefig(pdf, format='pdf') 
