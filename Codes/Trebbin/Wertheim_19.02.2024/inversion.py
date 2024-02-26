#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:43:24 2023

@author: sadegh
"""
# %% import lberaries
import numpy as np
import utm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pygimli.frameworks import harmfit
from sklearn.ensemble import IsolationForest
from matplotlib.colors import LogNorm
import geopandas as gpd
import fiona
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
# %% import data

## Define file paths and farm names
# filepath = 'Geophilus_aufbereitet_2023-04-05_New.csv'
# farmName = 'aufbereitet'

filepath = 'TRB_Wertheim_Geophilus_roh_221125.csv'
farmName = 'Trebbin Wertheim'

# filepath = 'Aussenschlag_2023_10_11.csv'
# farmName_Original = 'Trebbin Aussenschlag'

# filepath_Original = 'Geophilus_aufbereitet_2024-01-30_11-23-14.txt'
# farmName_Original = 'Großbeeren_1'

# filepath = 'BFD_Banane_Geophilus_221118_roh.csv'
# farmName = 'Beerfelde Banane'

# filepath = 'BFD_Fettke_Geophilus_roh.csv'
# farmName = 'Beerfelde Fettke'

# farmName_Processed = 'Beerfelde_Fettke_Processed data'
# filepath_original = 'Beerfelde 2022_10_21-original.csv'
# filepath_Original = "BFD_Fettke_Geophilus_roh.csv" # Origial data
# farmName_Original = 'Beerfelde Fettke_Original'


# filepath = 'BDW_Hoben_Geophilus_221020_roh.csv'
# farmName = 'Brodowin Hoben'

# filepath = 'BDW_Seeschlag_Geophilus_roh_221020 - Copy.csv'
# farmName = 'Brodowin Seeschlag'

# filepath = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName = 'Boossen_1211'

# filepath_Processed = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1601'


# %% Forward Operator and response function
from pygimli.frameworks import Modelling, Inversion
from pygimli.viewer.mpl import drawModel1D
import pygimli as pg


"""VES inversion."""

class VESRhoModelling(Modelling):  # 1D
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


class VES2dModelling(pg.frameworks.MeshModelling):
    """2D ... LCI."""
    def __init__(self, thk, nS, **kwargs):
        super().__init__()
        self.nL = len(thk) + 1
        self.nS = nS
        self.grid = pg.createGrid(self.nL+1, self.nS+1)
        self.setMesh(self.grid)
        self.fop1d = [VESRhoModelling(thk, **kwargs) for i in range(self.nS)]
        self.J = pg.BlockMatrix()
        resp1 = self.fop1d[0].response(pg.Vector(len(thk)+1, 1.0))
        #resp1 = self.fop1d[0].response(pg.Vector(len(thk)+1))

        nd = len(resp1)
        for i, fop in enumerate(self.fop1d):
            self.J.addMatrix(fop.jacobian(), i*nd, i*self.nL)

        self.J.recalcMatrixSize()
        self.setJacobian(self.J)
    
    def response(self, model):
        mod = np.reshape(model, [self.nS, self.nL])
        return np.concatenate(
            [self.fop1d[i].response(mod[i]) for i in range(self.nS)])
    
    def createJacobian(self, model):
        mod = np.reshape(model, [self.nS, self.nL])
        for i in range(self.nS):
            self.fop1d[i].createJacobian(mod[i])
        
        self.setJacobian(self.J)
        
#%% 1D inversion on the selected line (fn)
#from Data_Processing import read_data_Aus_lineIndexed
def extract_data_from_selected_line(filepath, line_number, amVec):
    """
    Process data, detect lines, and select data points for the specified line.

    Parameters:
    - data_file: Path to the CSV data file.
    - max_angle_deviation: Maximum angle deviation for line detection.
    - line_number: Index of the selected line.
    - amVec: Vector of apparent resistivity values.

    Returns:
    - selected_line_data: Numpy array containing data for the selected line.
    """
    data = read_data_Aus_lineIndexed(filepath)
    data_filtered = data[data[:, -1] != 0] # remove individual points
    selected_lines = np.unique(data_filtered[:,-1])
      
    line_number = np.asarray(line_number)

    # Get the line number of the selected line
    selected_line_number = selected_lines[line_number]

    # Create a mask to filter the data points belonging to the selected line
    line_mask = (data_filtered[:,-1] == selected_line_number)

    # Use the mask to select the data points of the selected line from processed_data
    selected_line_data = data_filtered[line_mask]
    print("Selected line: ", line_number)
    print("Selected line size: ", selected_line_data.shape)
    
    return selected_line_data



def inversion_1D_line(selected_line_data, amVec, farmName, line_number):
    """
    Perform 1D inversion on the given line data.

    Parameters:
    - selected_line_data: Numpy array containing data for the selected line.
    - amVec: Vector of apparent resistivity values.
    - farmName: Name of the farm or site for saving results.
    - line_numbers: List of line numbers to include in the filename.

    Returns:
    - chi2_indiv: List of chi-square values for individual inversions.
    """
    # Initialize ABMN values based on amVec
    b = 1.0
    bmVec = np.sqrt(amVec**2 + b**2)

    # Model space
    thk = np.ones(15) * 0.1
    
    # Initialize the DC 1D Forward Modelling Operator
    fop = VESRhoModelling(thk, am=amVec, an=bmVec, bm=bmVec, bn=amVec)

    chi2_indiv = []
    
    for data in selected_line_data:
        # Extract the Easting and Northing columns from the data
        #Eutm, Nutm = data[0], data[1]
        
        # Extract the resistivity values (Rho) for different spacings from the data
        Rho1, Rho2, Rho3, Rho4, Rho5 = data[3], data[4], data[5], data[6], data[7]
        
        # Initialize empty lists to store the inversion results
        StmodelsMean = []   # Stores the mean inversion models for each reference point
        chi2Vec_mean = []   # Stores the chi-square values for the mean inversion of each reference point
        chi2Vec_indiv = []  # Stores the chi-square values for the individual inversions of each reference point
        chi2_indiv = []     # Stores the individual inversion models for each reference point
        
        num_valid_points_list = []  # List to store the number of valid points for each reference point
        skipped_points_list = []    # List to store the skipped points for each reference point
        
     
        mydata = selected_line_data[:, 3:8]
        Stmodels = []
        skipped_points = []  # Create an empty list to keep track of skipped points
        inv_response_all_soundings = []
        # Create an error matrix for individual data
        error_replaced_individual = np.ones_like(mydata) * 0.03
        # Replace NaN values with 1000
        mydata[np.isnan(mydata)] = 1000
        # Identify the indices where data is replaced with 1000
        indices_to_replace_individual = np.where(mydata == 1000)
        # Set error to 10000 for replaced values in both vectors
        error_replaced_individual[indices_to_replace_individual] = 10000
        
        # Inversion of individual 
        for i, indivData in enumerate(mydata):
            if not np.isnan(indivData).any():
                # Check if indivData contains NaN values
                inv_indiv = Inversion(fop=fop)  # Passing the fwd operator in the inversion
                inv_indiv.setRegularization(cType=1)  # cType=1 for smoothness constraint
                modelInd = inv_indiv.run(indivData, error_replaced_individual[i,:], lam=350, zWeight=3, lambdaFactor=1.12, startModel=300, verbose=True)           
                Stmodels.append(modelInd)
                np.savetxt(f'{farmName_Original}_1DInv_line_number_{line_number}.txt', Stmodels, delimiter=';', fmt='%s')
                chi2 = inv_indiv.inv.chi2()
                chi2Vec_indiv.append(chi2)
                # forward model response
                inv_response = inv_indiv.response
                inv_response_all_soundings.append(inv_response)

            else:
                skipped_points.append(selected_line_data[i])  # Add the index of the skipped point to the list
        chi2_indiv.append(np.array(chi2Vec_indiv))
        np.savetxt(f'{farmName_Original}_chi2_indiv-1dInv_line_number_{line_number}.csv', chi2_indiv, delimiter=';', fmt='%s')
        # save forward model response
        inv_response_filename = f'{farmName_Original}_1Dinv_response_line_number {line_number}.txt'
        np.savetxt(inv_response_filename, np.array(inv_response_all_soundings), delimiter=';')
        return Stmodels, chi2_indiv

# Example usage:


 # read Processed data
# filepath_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName_Processed = 'Trebbin Wertheim_Processed'
# filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName = 'Boossen_1601'
# filepath_Processed = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1211'
# filepath_Original = "Beerfelde 2022_10_21-original.csv"  # Original data
# farmName_Original = 'Beerfelde Fettke_Original'
# filepath_Processed = 'BFD_Fettke_Geophilus_roh_Processed.csv'
# farmName_Processed = 'Beerfelde Fettke_Processed'

# # Read harmfit data
# harmfitted_data_file = f"harmfit_on_original_{farmName_Original}.txt"
# harmfit_data = read_data_UTM(harmfitted_data_file)
# # read original data
# original_data = read_data("Beerfelde 2022_10_21-original.csv")
# downsampled_data = original_data[::2]
# # Read Smoothed harmfit data
# smoothed_harmfitted_data_file = f"smoothed_harmfit_{farmName_Original}.txt"
# smoothedharmfit_data = read_data_UTM(smoothed_harmfitted_data_file)

# filepath_Original = 'Aussenschlag_2023_10_11.csv'
# farmName_Original = 'Trebbin Aussenschlag'

filepath_Original = 'Geophilus_aufbereitet_2024-01-30_11-23-14.txt'
farmName_Original = 'Großbeeren_1'

data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

# Define a hypothetical ref point
refEutm, refNutm = [385489, 5801147]
refName = 1
# Stack the arrays horizontally
refPoints = np.column_stack([refName, refEutm, refNutm])
# Define header and stack header with ref_Points
header = ['Name', 'E', 'N']
refPointsTable = np.vstack((header, refPoints))
    
# farm_name = 'Trebbin Aussenschlag'
# refpoints = f'{farmName_Original} farm_reference_Points.csv'
# RefPoints = pd.read_csv(refpoints, delimiter=";")

line_number = 21
amVec = np.arange(1, 6) * 0.6  
# selected_line_data = extract_data_from_selected_line(data_lineIndexed_file, line_number=line_number, amVec=amVec)

# Stmodels, chi2 = inversion_1D_line(selected_line_data, amVec, farmName_Original, line_number)




#
#%% plot 1D inversion result for one line (fn)


def plot_1d_inversion_results(farmName, line_number, selected_line_data, RefPoints, thk, chi2, inv_result):

    with PdfPages(f'{farmName_Original}_1D_Inversion_line_number_{line_number}.pdf') as pdf:
        Inv_indiv = np.loadtxt(f'{farmName_Original}_1DInv_line_number_{line_number}.txt', delimiter=';')
        chi2Indiv = np.array(chi2)
        mydata = selected_line_data[:, 3:8]
        Eutm, Nutm = selected_line_data[:, 0], selected_line_data[:, 1]

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15.5, 8), gridspec_kw={'height_ratios': [1, 1]})

        fig.suptitle(f'1D_Inversion (line_number_{line_number}, Farm Name: {farmName_Original})', fontsize=16)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        ax0, ax1, ax2, ax3, ax4, ax5 = axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]

        ax0.bar(range(len(chi2Indiv[0])), chi2Indiv[0], width=0.7, align='center', label=f"chi2 for individual data")
        ax0.grid(True)
        ax0.set_xlim([-0.5, len(chi2Indiv[0]) - 0.5])
        ax0.legend(fontsize=8, loc=4)
        ax0.set_xlabel('individual data')
        ax0.set_ylabel('$\u03C7^2$')
        ax0.set_title( f'Chi-Square')
        ax0.set_yscale('log')

        # plot the forward model response to check if these it has a systematic missfit with data
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        inv_response = np.loadtxt(f'{farmName_Original}_1Dinv_response_line_number {line_number}.txt', delimiter=';')
        x = np.arange(mydata.T.shape[1])
        y = np.arange(mydata.T.shape[0])
        extent = [x[0], x[-1], y[0], y[-1]]
        norm = colors.LogNorm(vmin=10, vmax=1000)
        im = ax1.imshow(inv_response.T, cmap='Spectral_r', norm=norm, extent=extent, interpolation="nearest")
        ax1.set_aspect('auto')  
        ax1.set_title('forward model response')
        ax1.set_yticklabels(y[::-1])
        ax1.set_ylabel('spacing')
        # Add a color bar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="9%", pad=0.5)  # Adjust pad as needed
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        
        ax2.set_xlim([5, 3000])
        ax2.legend(fontsize=8, loc=2)
        ax2.set_title('Model')
        for inv in Inv_indiv: 
            drawModel1D(ax2, thickness=thk, values=inv, plot='semilogx', color='lightgray', linewidth=1)

        ax3.set_xlim([min(Eutm), max(Eutm)])
        ax3.set_ylim([min(Nutm), max(Nutm)])
        ax3.axis("equal")
        ax3.plot(Eutm, Nutm, "o", markersize=4, mew=1, label=f'line {line_number}')
        # Plot the reference points and add labels
        # for i, (e, n, label) in enumerate(zip(RefPoints['E'], RefPoints['N'], RefPoints['Name'])):
        #     ax3.plot(e, n, "x", markersize=8, mew=2)
        #     ax3.text(e, n, label, ha='right', va='bottom', fontsize=8)
        #ax3.plot(RefPoints['E'], RefPoints['N'], "o", markersize=8, mew=2, label='Reference Point')
        ax3.set_title(f'Reference point and its nearest line')
        ax3.legend(prop={'size': 8})
        ax3.set_xlabel('easting')
        ax3.set_ylabel('northing')
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.xaxis.set_tick_params(rotation=30)
        ax3.grid()
       
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        norm = colors.LogNorm(vmin=10, vmax=1000)
        # Create a meshgrid to specify coordinates for the image
        x = np.arange(mydata.T.shape[1])
        y = np.arange(mydata.T.shape[0])
        extent = [x[0], x[-1], y[0], y[-1]]
        # Display the numerical values with the color map
        im = ax4.imshow(mydata.T, cmap='Spectral_r', norm=norm, extent=extent, interpolation="nearest")
        ax4.set_aspect('auto')  
        ax4.set_title('Rhoa')
        ax4.set_yticklabels(y[::-1])
        ax4.set_ylabel('spacing')
        # Add a color bar
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("bottom", size="9%", pad=0.5)  
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
       
        plt.tight_layout()  

        pg.viewer.mpl.showStitchedModels(Inv_indiv, ax=ax5, x=None, cMin=10, cMax=1000, cMap='Spectral_r', thk=thk, logScale=True, title='Model (Ohm.m)', zMin=0, zMax=0, zLog=False)

        plt.show()

        # Save the plot to the PDF file
        pdf.savefig(fig)
        fig.savefig(f'{farmName_Original}_1D_Inversion_line_number_{line_number}.jpg', format='jpg', dpi=300)  # This saves as a JPEG

        plt.close()

# Example usage:
# farmName = 'Trebbin Aussenschlag'
# farmName = 'Beerfelde Fettke'

# filepath_Original = "Beerfelde 2022_10_21-original.csv"  # Original data
# farmName_Original = 'Beerfelde Fettke_Original'
# filepath_Processed = 'BFD_Fettke_Geophilus_roh_Processed.csv'
# farmName_Processed = 'Beerfelde Fettke_Processed'

 # read Processed data
filepath_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
farmName_Processed = 'Trebbin Wertheim_Processed'
# filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName = 'Boossen_1601'
# # Read harmfit data
# harmfitted_data_file = f"harmfit_on_original_{farmName_Original}.txt"

# #harmfit = f'harmfit_on_original_{farmName}.txt'
# #data_file =harmfit

# smoothed_harmfitted_data_file = f"smoothed_harmfit_{farmName_Original}.txt"
# smoothedharmfit_data = read_data_UTM(smoothed_harmfitted_data_file)

# filepath_Original = 'Aussenschlag_2023_10_11.csv'
# farmName_Original = 'Trebbin Aussenschlag'
# filepath_Original = 'Geophilus_aufbereitet_2024-01-30_11-23-14.txt'
# farmName_Original = 'Großbeeren_1'

data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

# Define a hypothetical ref point
refEutm, refNutm = [385489, 5801147]
refName = 1
# Stack the arrays horizontally
refPoints = np.column_stack([refName, refEutm, refNutm])
# Define header and stack header with ref_Points
header = ['Name', 'E', 'N']
refPointsTable = np.vstack((header, refPoints))

data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

line_number = 21

#chi2 = pd.read_csv(f'{farmName_Original}_chi2_indiv-1dInv_line_number_{line_number}.csv', delimiter=';', header=None)
# inv_result = np.loadtxt(f'{farmName_Original}_1DInv_line_number_{line_number}.txt', delimiter=';')
refpoints = f'{farmName_Original} farm_reference_Points.csv'
#RefPoints = pd.read_csv(refpoints, delimiter=";")

# # selected_line_data = extract_data_from_selected_line(data_file, max_angle_deviation, line_number=line_number, amVec=np.arange(1, 6) * 0.6)
thk = np.ones(15) * 0.1
# selected_line_data = extract_data_from_selected_line(data_lineIndexed_file, line_number=line_number, amVec=amVec)

# plot_1d_inversion_results(filepath_Original, line_number, selected_line_data, refPoints, thk, chi2, inv_result)


#%% 2D inversion for all lines (.)needs running)
if 0:
    def inversion_2D_all_lines(data_file, farmName, max_angle_deviation, amVec, bmVec, nLayer):
        # Read the original data
        original_data = read_original_data(data_file)
    
        # Extract line data based on the specified maximum angle deviation
        line_detector = LineDetector(maxDeviation=maxDeviation)
        line_indices = line_detector.detect_lines(original_data[:, 0], original_data[:, 1])
    
        # Get unique line indices
        unique_lines = np.unique(line_indices)
    
        # Iterate through each line and perform 2D inversion
        for line_number in unique_lines:
            selected_line_data = extract_data_from_selected_line(data_file, max_angle_deviation, line_number, amVec)
            
            
            nSoundings = len(selected_line_data)
            thk = np.ones(15) * 0.1
            selected_line_data = selected_line_data
            error_replaced_individual = np.ones_like(selected_line_data[:, 3:8]) * 0.03
            selected_line_data[:, 3:8][np.isnan(selected_line_data[:, 3:8])] = 1000
            indices_to_replace_individual = np.where(selected_line_data[:, 3:8] == 1000)
            error_replaced_individual[indices_to_replace_individual] = 10000
            fop2d = VES2dModelling(thk=thk, nS=nSoundings, am=amVec, an=bmVec, bm=bmVec, bn=amVec)
            inv_2d = Inversion(fop=fop2d)
            inv_2d.setRegularization(cType=1)
            selected_data_flat = selected_line_data[:, 3:8].flatten()
            error_replaced_individual_flat = error_replaced_individual.flatten()
            models_2d = inv_2d.run(selected_data_flat, error_replaced_individual[0], lam=20, lambdaFactor=0.9, startModel=300, verbose=True)
            np.savetxt(f'{farmName}_2DInv_line_number {line_number}.csv', models_2d, delimiter=';', fmt='%s')
            models_reshaped = np.asarray(models_2d).reshape((nSoundings, nLayer))
            model_filename = f'{farmName}_2D-InvResult_line_number {line_number}.txt'
            np.savetxt(model_filename, models_reshaped, delimiter='\t')
            
            # Calculate chi-square values for each data point
            inv_response = np.reshape(inv_2d.response, [-1, 5])
            data = selected_line_data[:, 3:8]
            chiSquareVec = []
            for i in range(len(inv_response)):
                chiSquare = np.mean((data[i, :] - inv_response[i, :]) ** 2 / (error_replaced_individual[i, :]) ** 2)
               # chiSquare = np.mean(((np.log(data[i]) - np.log(inv_response[i])) / error[i])** 2)
    
                chiSquareVec.append(chiSquare)
                chi2_filename = f'{farmName}_chi2_2DInv_line_number {line_number}.csv'
                np.savetxt(chi2_filename, np.array(chiSquareVec), delimiter=';')
    
                inv_response_filename = f'{farmName}_2DInvResponse_line_number {line_number}.csv'
                np.savetxt(inv_response_filename, inv_response, delimiter=';')
    
    # Example usage:
    data_file = 'TRB_Wertheim_Geophilus_roh_221125.csv'
    farmName = 'Trebbin Wertheim'
    max_angle_deviation = 22.0
    amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
    bmVec = amVec  # Adjust bmVec according to your data
    nLayer = 15  # Adjust the number of layers as needed
    
    inversion_2D_all_lines(data_file, farmName, max_angle_deviation, amVec, bmVec, nLayer)

#%% 1D Inversion for all lines 
import numpy as np
from pygimli.frameworks import Modelling, Inversion
from pygimli.viewer.mpl import drawModel1D


def inversion_1D_all_lines(filepath, amVec, farmName):
    """
    Perform 1D inversion for all selected lines' data.

    Parameters:
    - selected_lines_data: List of numpy arrays containing data for each selected line.
    - amVec: Vector of apparent resistivity values.
    - farmName: Name of the farm or site for saving results.

    Returns:
    - chi2_indiv_list: List of lists of chi-square values for individual inversions of each line.
    - Stmodels_list: List of lists of inversion models for each line.
    """
    spacing_labels = ['S1', 'S2', 'S3', 'S4', 'S5']

    data = read_data_lineIndexed(filepath)
    data_no_nan = data[~np.isnan(data).any(axis=1)]
    data_filtered = data_no_nan[data_no_nan[:, -1] != 0] # remove individual points
    selected_lines = np.unique(data_filtered[:,-1])

    
    # Initialize ABMN values based on amVec
    b = 1.0
    bmVec = np.sqrt(amVec**2 + b**2)

    # Model space
    thk = np.ones(15) * 0.1

    # Initialize the DC 1D Forward Modelling Operator
    fop = VESRhoModelling(thk, am=amVec, an=bmVec, bm=bmVec, bn=amVec)

    chi2_indiv_list = []
    Stmodels_list = []
    invResult_AllLines = []
    forward_resp_AllLines = []
    for line_idx, line_num in enumerate(selected_lines, start=1):
        point_indices_of_interest = np.where(data_filtered[:,-1] == line_num)[0]
        data_for_inv = data_filtered[point_indices_of_interest]
        error = np.ones_like(data_for_inv[1]) * 0.03
        
        chi2Vec_indiv = []
        Stmodels = []
        results_list = []
        inv_response_all_soundings = []
        xy_coord_currentLine = []

        # Create an error matrix for individual data
        error = np.ones_like(data_for_inv[:, 3:8][0]) * 0.03
        for i, indivData in enumerate(data_for_inv):
            inv_indiv = Inversion(fop=fop)  # Passing the fwd operator in the inversion
            inv_indiv.setRegularization(cType=1)  # cType=1 for smoothness constraint
            modelInd = inv_indiv.run(indivData[3:8], error, lam=350, zWeight=3, lambdaFactor=1.12, startModel=300, verbose=True)
            Stmodels.append(modelInd)

            xy_coord = data_for_inv[i, 0:2]  # Extract x and y coordinates
            xy_coord_currentLine.append(xy_coord)
            chi2 = inv_indiv.inv.chi2()
            chi2Vec_indiv.append(chi2)
            line_index = np.array(data_for_inv[:,-1])
            # Append the results for the current point
            results_list.append(np.hstack((xy_coord, modelInd, chi2)))
            
            # forward model response
            inv_response = inv_indiv.response
            inv_response_all_soundings.append(inv_response)
      
        # making an array for the inversion result for all data
        invResult_currentLine = np.hstack((np.array(xy_coord_currentLine), np.array(Stmodels), 
                                           np.reshape(chi2Vec_indiv, (-1, 1)), np.reshape(line_index, (-1, 1))))
        invResult_AllLines.append(invResult_currentLine)
        invResult_combined = np.concatenate(invResult_AllLines, axis=0)
        header = "x; y; " + "; ".join([f"inv{k + 1}" for k in range(len(modelInd))]) + "; chi2;  line_index"
        result_file = f'{farmName}_1Dinv_response_all_lines_combined.txt'
        np.savetxt(result_file, invResult_combined, delimiter='\t', fmt='%.3f', header=header)

        # making an array for the forward response for all data    
        forward_resp_currentLine = np.hstack((np.array(xy_coord_currentLine), np.array(inv_response_all_soundings), 
                                           np.reshape(line_index, (-1, 1))))
        forward_resp_AllLines.append(forward_resp_currentLine)
        forward_resp_combined = np.concatenate(forward_resp_AllLines, axis=0)
        header_forward = "x; y; " + "; ".join([f"ForwResp{k + 1}" for k in range(5)]) + ";  line_index"
        result_file_forward = f'{farmName}_1Dforward_response_all_lines_combined.txt'
        np.savetxt(result_file_forward, forward_resp_combined, delimiter='\t', fmt='%.3f', header=header_forward)
        
        chi2_indiv_list.append(chi2Vec_indiv)
        Stmodels_list.append(Stmodels)

    return Stmodels_list, chi2_indiv_list


def read_data_lineIndexed(filepath):
    EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma, line_index = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
    # EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, nan_data, Rho5, Gamma, BFI, lfd, Datum, Zeit = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
    data = np.column_stack((EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma, line_index))
    return data


# Example usage:
# data_file = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName = 'Trebbin Wertheim'

filepath_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
farmName_Processed = 'Trebbin Wertheim_Processed'

# filepath_Processed = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1601'
# filepath_Processed = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1211'
# filepath_Original = 'Aussenschlag_2023_10_11.csv'
# farmName_Original = 'Trebbin Aussenschlag'

# filepath_Original = 'Geophilus_aufbereitet_2024-01-30_11-23-14.txt'
# farmName_Original = 'Großbeeren_1'

# filepath_Original = "BFD_Fettke_Geophilus_roh.csv" # Origial data
# farmName_Original = 'Beerfelde Fettke_Original'

data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

# # Define a hypothetical ref point
# refEutm, refNutm = [385489, 5801147]
# refName = 1
# # Stack the arrays horizontally
# refPoints = np.column_stack([refName, refEutm, refNutm])
# # Define header and stack header with ref_Points
# header = ['Name', 'E', 'N']
# refPointsTable = np.vstack((header, refPoints))

kmlFile = 'TRB_Wertheim_Referenzpunkte_gemessen_2022-12-01.kml'
fiona.drvsupport.supported_drivers['KML'] = 'rw'
gdf = gpd.read_file(kmlFile)
refLat = np.array(gdf['geometry'].y)
refLon = np.array(gdf['geometry'].x)
refEutm, refNutm, refzone, refletter = utm.from_latlon(refLat, refLon)
refName = np.array(gdf['Name'])
ref_Points = np.column_stack([refName, refEutm, refNutm])
header = ['Name', 'E', 'N']
refPointsTable = np.vstack((header, ref_Points))
 
#  inv for original data
amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
# Stmodels, chi2 = inversion_1D_all_lines(data_lineIndexed_file, amVec, farmName_Original)

#  inv for processed (HLOF)
farmName_Processed = 'Trebbin Wertheim_HLOF'

# farmName_Original = 'Großbeeren_1'
data_type_HLOF = 'LOF'
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName_Original}.txt"
#Stmodels, chi2 = inversion_1D_all_lines(filepath_Processed, amVec, farmName_Processed)

#%% 1D inversion plotting for all lines 

def plot_1d_inversion_results_allLines(filepath, amVec, farmName, data_type):
    data = read_data_lineIndexed(filepath)
    data_no_nan = data[~np.isnan(data).any(axis=1)]
    data_filtered = data_no_nan[data_no_nan[:, -1] != 0] # remove individual points
    selected_lines = np.unique(data_filtered[:,-1])
    line_data = {}  # Dictionary to hold data for each line

    with PdfPages(f'{farmName}_1D_Inversion_all_lines_ {data_type}.pdf') as pdf:
        for line_num in selected_lines:
            line_mask = (data_filtered[:, -1] == line_num)
            line_data[line_num] = data_filtered[line_mask][:, 3:8]
            mydata = np.asarray(line_data[line_num])
            
            # Load the inversion results from the saved file
            inv_result_data = np.loadtxt(f'{farmName}_1Dinv_response_all_lines_combined.txt', delimiter='\t', skiprows=1)
            # Extract relevant information
            x_values = inv_result_data[:, 0]
            y_values = inv_result_data[:, 1]
            inversion_results = inv_result_data[:, 2:-2]  # Exclude x, y, chi2, and line_index columns
            chi2_values = inv_result_data[:, -2]
            line_index_values = inv_result_data[:, -1]

            # extract inv result for the current line
            inv_result_data_current_line = inv_result_data[inv_result_data[:, -1] == line_num]
            Inv_indiv = inv_result_data_current_line[:, 2:18]
            chi2 = inv_result_data_current_line[:, -2]
            chi2Indiv = np.array(chi2.T)

            Eutm, Nutm = data_filtered[line_mask][:, 0] , data_filtered[line_mask][:, 1]
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15.5, 8), gridspec_kw={'height_ratios': [1, 1]})
    
            fig.suptitle(f'1D_Inversion (line_{int(line_num)}, Farm Name: {farmName_Original} _ {data_type})', fontsize=16)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
            ax0, ax1, ax2, ax3, ax4, ax5 = axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]
            
            
            ax0.set_xlim([min(Eutm), max(Eutm)])
            ax0.set_ylim([min(Nutm), max(Nutm)])
            ax0.axis("equal")
            ax0.plot(Eutm, Nutm, "o", markersize=4, mew=1, label=f'line {int(line_num)}')
            ax0.set_title(f'Reference point and its nearest line')
            ax0.legend(prop={'size': 8})
            ax0.set_xlabel('easting')
            ax0.set_ylabel('northing')
            ax0.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax0.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax0.xaxis.set_tick_params(rotation=30)
            ax0.grid()
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            norm = colors.LogNorm(vmin=10, vmax=1000)
            # Create a meshgrid to specify coordinates for the image
            x = np.arange(mydata.shape[0])
            y = np.arange(mydata.shape[1])
            y_ticks = y[::-1]
            extent = [x[0], x[-1], y[0], y[-1]]
            # Display the numerical values with the color map
            #im = ax4.imshow(mydata, cmap='Spectral_r', norm=norm, extent=extent)
            im = ax1.imshow(mydata.T, cmap='Spectral_r', norm=norm, extent=extent, interpolation="nearest")
            ax1.set_aspect('auto')  
            ax1.set_title('Rhoa')
            ax1.set_yticks(y)
            ax1.set_yticklabels(y[::-1])
            ax1.set_ylabel('spacing')
            # Add a color bar
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("bottom", size="9%", pad=0.5)  
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            
            # plot the forward model response to check if these it has a systematic missfit with data
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            forward_resp = np.loadtxt(f'{farmName}_1Dforward_response_all_lines_combined.txt', delimiter='\t', skiprows=1)
            forward_resp_current_line = forward_resp[forward_resp[:, -1] == line_num][:, 2:-1]          
            inv_response = Inv_indiv
            x = np.arange(mydata.T.shape[1])
            y = np.arange(mydata.T.shape[0])
            extent = [x[0], x[-1], y[0], y[-1]]
            norm = colors.LogNorm(vmin=10, vmax=1000)
            im = ax2.imshow(forward_resp_current_line.T, cmap='Spectral_r', norm=norm, extent=extent, interpolation="nearest")
            ax2.set_aspect('auto')  
            ax2.set_title('forward model response')
            ax2.set_yticklabels(y[::-1])
            ax2.set_ylabel('spacing')
            # Add a color bar
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("bottom", size="9%", pad=0.5)  # Adjust pad as needed
            
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')   
            ax3.bar(range(len(chi2Indiv)), chi2Indiv, width=0.7, align='center', label=f"chi2 for individual data")
            ax3.grid(True)
            ax3.set_xlim([-0.5, len(chi2Indiv) - 0.5])
            ax3.legend(fontsize=8, loc=4)
            ax3.set_xlabel('individual data')
            ax3.set_ylabel('$\u03C7^2$')
            ax3.set_title( f'Chi-Square')
            ax3.set_yscale('log')
            
            ax4.set_xlim([5, 3000])
            ax4.legend(fontsize=8, loc=2)
            ax4.set_title('Model')
            for inv in Inv_indiv: 
                drawModel1D(ax4, thickness=thk, values=inv, plot='semilogx', color='lightgray', linewidth=1)

            plt.tight_layout()  
    
            pg.viewer.mpl.showStitchedModels(Inv_indiv, ax=ax5, x=None, cMin=10, cMax=1000, cMap='Spectral_r', thk=thk, logScale=True, title='Model (Ohm.m)', zMin=0, zMax=0, zLog=False)
            
            plt.show()
    
            # Save the plot to the PDF file
            pdf.savefig(fig)
            plt.close()
    

# Example usage:
# filepath_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName_Processed = 'Trebbin Wertheim_Processed'

amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
thk = np.ones(15) * 0.1

# filepath_Processed = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1601'
# filepath_Processed = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1211'

# filepath_Original = 'Aussenschlag_2023_10_11.csv'
# farmName_Original = 'Trebbin Aussenschlag'

# filepath_Original = 'Geophilus_aufbereitet_2024-01-30_11-23-14.txt'
# farmName_Original = 'Großbeeren_1'
# filepath_Original = "BFD_Fettke_Geophilus_roh.csv" # Origial data
# farmName_Original = 'Beerfelde Fettke_Original'

filepath_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
farmName_Original = 'Trebbin Wertheim'

data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"


kmlFile = 'TRB_Wertheim_Referenzpunkte_gemessen_2022-12-01.kml'
fiona.drvsupport.supported_drivers['KML'] = 'rw'
gdf = gpd.read_file(kmlFile)
refLat = np.array(gdf['geometry'].y)
refLon = np.array(gdf['geometry'].x)
refEutm, refNutm, refzone, refletter = utm.from_latlon(refLat, refLon)
refName = np.array(gdf['Name'])
ref_Points = np.column_stack([refName, refEutm, refNutm])
header = ['Name', 'E', 'N']
refPointsTable = np.vstack((header, ref_Points))
 

# # Define a hypothetical ref point
# refEutm, refNutm = [385489, 5801147]
# refName = 1
# # Stack the arrays horizontally
# refPoints = np.column_stack([refName, refEutm, refNutm])
# # Define header and stack header with ref_Points
# header = ['Name', 'E', 'N']
# refPointsTable = np.vstack((header, refPoints))


# inv plot for original data
data_type_org = 'Original'
# plot_1d_inversion_results_allLines(data_lineIndexed_file, amVec, farmName_Original, data_type_org)


# inv plot for processed (HLOF)
farmName_Processed = 'Trebbin Wertheim_HLOF'
farmName_Original = 'Trebbin Wertheim'
data_type_HLOF = 'LOF'
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName_Original}.txt"
plot_1d_inversion_results_allLines(filepath_Processed, amVec, farmName_Processed, data_type_HLOF)

#%% inversion result subplot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

farmName_Processed = 'Trebbin Wertheim_HLOF'

# Load data directly from the file using np.genfromtxt
inv_res = np.genfromtxt(f'{farmName_Processed}_1Dinv_response_all_lines_combined.txt', delimiter='\t', skip_header=1)

#fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(15, 8))
x_offset = np.min(inv_res[:, 0])
y_offset = np.min(inv_res[:, 1])
fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(11, 5), 
                       gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

norm = LogNorm(10, 1000)
for i, a in enumerate(ax.flat):
    rho = inv_res[:, i+2]  # Assuming the rho values start from the third column
    im = a.scatter(inv_res_filtered[:, 0] - x_offset, inv_res_filtered[:, 1] - y_offset, 
               c=rho, s=0.1, norm=norm, cmap="Spectral_r", alpha=1)
    #im = a.scatter(inv_res[:, 0]-383500, inv_res[:, 1]-579750, c=rho, s=0.1, norm=norm, cmap="Spectral_r", alpha=1)
    a.set_aspect(1.0)
    a.set_title("z={:.1f}-{:.1f}m".format(i*0.1, (i+1)*0.1))

plt.tight_layout(rect=[0.03, 0.03, 0.93, 0.95])  # Adjust the layout to make room for the suptitle

# Add a color bar
cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Resistivity (Ohm.m)')

# Set common labels
fig.text(0.5, 0.02, 'Easting', ha='center', va='center')
fig.text(0.05, 0.5, 'Northing', ha='center', va='center', rotation='vertical')

plt.suptitle(f'{farmName_Processed} Inversion Results', fontsize=16)
# Save the plot as PDF
plt.savefig(f'{farmName_Processed}_subplot_inversion_results.pdf')
# Save the plot as JPEG
plt.savefig(f'{farmName_Processed}_subplot_inversion_results.jpg', dpi=300)

plt.show()

#%% inversion result subplot chi2<10

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

farmName_Processed = 'Trebbin Wertheim_HLOF'

# Load data directly from the file using np.genfromtxt
inv_res = np.genfromtxt(f'{farmName_Processed}_1Dinv_response_all_lines_combined.txt', delimiter='\t', skip_header=1)
chi2Lim = 20
inv_res_filtered = inv_res[inv_res[:, -2] <= chi2Lim] ## Filter out rows where the Chi2 is greater than 100

# Save the filtered inversion results to a text file
inv_res_with_chi2andlineInx = np.genfromtxt(f'{farmName_Processed}_1Dinv_response_all_lines_combined.txt', delimiter='\t', skip_header=0)
inv_res_with_chi2andlineInx_filtered = inv_res[inv_res[:, -2] <= chi2Lim] ## Filter out rows where the Chi2 is greater than 100
header_inv = 'x; y; ' + "; ".join([f'inv{i}' for i in range(1, inv_res_filtered.shape[1]-3)]) + "; chi2; line_index"
np.savetxt(f'{farmName_Processed}_1Dinv_response_all_lines_combined_chi2_less_than_{chi2Lim}.txt', 
           inv_res_filtered, delimiter='\t', header=header_inv, fmt='%.3f')
fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(11, 5), 
                       gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

norm = LogNorm(10, 1000)
for i, a in enumerate(ax.flat):
    rho = inv_res_filtered[:, i+2]  # Assuming the rho values start from the third column
    im = a.scatter(inv_res_filtered[:, 0]-383500, inv_res_filtered[:, 1]-579750, c=rho, s=0.1, norm=norm, cmap="Spectral_r", alpha=1)
    a.set_aspect(1.0)
    a.set_title("z={:.1f}-{:.1f}m".format(i*0.1, (i+1)*0.1))
plt.tight_layout(rect=[0.03, 0.03, 0.93, 0.95])  # Adjust the layout to make room for the suptitle

# Add a color bar
cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Resistivity (Ohm.m)')

# Set common labels
fig.text(0.5, 0.02, 'Easting', ha='center', va='center')
fig.text(0.05, 0.5, 'Northing', ha='center', va='center', rotation='vertical')

plt.suptitle(f'{farmName_Processed} Inversion Results, chi2<{chi2Lim}', fontsize=16)
# Save the plot as PDF
plt.savefig(f'{farmName_Processed}_subplot_inversion_results_chi2,{chi2Lim}.pdf')
# Save the plot as JPEG
plt.savefig(f'{farmName_Processed}_subplot_inversion_results_chi2,{chi2Lim}.jpg', dpi=300)

plt.show()
#%% inversion result multipage pdf

farmName_Processed = 'Trebbin Wertheim_HLOF'

# Load data directly from the file using np.genfromtxt
inv_res = np.genfromtxt(f'{farmName_Processed}_1Dinv_response_all_lines_combined.txt', delimiter='\t', skip_header=1)
#inv_res_filtered = inv_res[inv_res[:, -2] <= 100] ## Filter out rows where the Chi2 is greater than 100

# Create a multi-page PDF file
pdf_filename = f'{farmName_Processed}_inversion_results_MPpdf.pdf'
with PdfPages(pdf_filename) as pdf:
    for i in range(15):  # Loop through all subplots
        fig, ax = plt.subplots(figsize=(8, 6))
        norm = LogNorm(10, 1000)
        rho = inv_res[:, i + 2]  # Assuming the rho values start from the third column
        im = ax.scatter(inv_res[:, 0] - 383500, inv_res[:, 1] - 579750, c=rho, s=0.4, norm=norm, cmap="Spectral_r", alpha=1)
        #ax.set_aspect(1.0)
        ax.set_title("z={:.1f}-{:.1f}m".format(i * 0.1, (i + 1) * 0.1))
        # Add color bar
        cbar = plt.colorbar(im, pad=0.02, shrink=0.8)
        cbar.set_label('Resistivity (Ohm.m)')
        # Set common labels
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        plt.suptitle(f'Inversion Results for {farmName_Processed}', fontsize=16)

        # Save the subplot into the PDF file
        pdf.savefig(fig)
        plt.show()

        plt.close(fig)  # Close the figure to release memory


#%% plot Chi2
farmName_Processed = 'Trebbin Wertheim_HLOF'

# Load data directly from the file using np.genfromtxt
inv_res = np.genfromtxt(f'{farmName_Processed}_1Dinv_response_all_lines_combined.txt', delimiter='\t', skip_header=1)
#inv_res_filtered = inv_res[inv_res[:, -2] <= 100] ## Filter out rows where the Chi2 is greater than 100

# Create a multi-page PDF file
pdf_filename = f'{farmName_Processed}_Chi2_inversion_results.pdf'
with PdfPages(pdf_filename) as pdf:
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = LogNorm(1, 300)
    chi2 = inv_res[:, -2]  # Assuming the rho values start from the third column
    im = ax.scatter(inv_res[:, 0] - 383500, inv_res[:, 1] - 579750, c=chi2, s=0.2, norm=norm, cmap="Spectral_r", alpha=1)
    ax.set_aspect(1.0)
    ax.set_title(r"$\chi^2$".format(i * 0.1, (i + 1) * 0.1))
    # Add color bar
    cbar = plt.colorbar(im, pad=0.02, shrink=0.5)
    cbar.set_label(r"$\chi^2$")
    # Set common labels
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    # Save the subplot into the PDF file
    plt.show()

    pdf.savefig(fig)
    plt.savefig(f'{farmName_Processed}_Chi2_inversion_results.jpg', dpi=300)

    plt.close(fig)  

#%% Geophilus Sounding Curves
import numpy as np
import matplotlib.pyplot as plt

# Given data
thk = np.ones(15) * 0.1
amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)

Rhos = np.array([[1200, 1500, 1710, 1850, 1990],
                 [1850, 1380, 1150, 1020, 950],
                 [2470, 2390, 2245, 2090, 1900],
                 [910, 930, 1000, 1080, 1180],
                 [930, 940, 1020, 1075, 1100],
                 [1260, 1390, 1360, 1290, 1200]])

colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# Plotting each sounding's rhoa values against thk
fig = plt.figure(figsize=(6, 6))
plt.title('Geophilus Sounding Curves', fontsize=13)
plt.xlabel('Apparent Resistivity (rhoa)')
plt.ylabel('amVec (m)')

for i, row in enumerate(Rhos):
    plt.plot(row, amVec, marker='o', color=colors[i], label=f'Model {i+1}')

plt.legend()
plt.grid(True)
plt.yticks(np.arange(min(amVec), max(amVec) + 0.6, 0.6))
plt.xscale('log')

# Invert y-axis
plt.gca().invert_yaxis()

plt.show()
fig.savefig('Geophilus_Sounding_Curves.jpg', format='jpg', dpi=300)


# %% Forward Operator and response function

amVec = np.arange(1, 6) * 0.6
b = 1.0
bmVec = np.sqrt(amVec**2 + b**2)
fop = VESRhoModelling(thk, am=amVec, an=bmVec, bm=bmVec, bn=amVec)
error = np.ones_like(np.array(Rhos)[0]) * 0.03
Stmodels = []

for Rho in Rhos:
    inv_indiv = Inversion(fop=fop)
    inv_indiv.setRegularization(cType=1)
    modelInd = inv_indiv.run(Rho, error, lam=350, zWeight=3, lambdaFactor=1.12, startModel=300, verbose=True)
    Stmodels.append(modelInd)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
fig.suptitle('Input Resistivity Models', fontsize=14)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
axes = axes.flatten()

subtitle_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
plot_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

for i, (ax, inv) in enumerate(zip(axes, Stmodels)):
    ax.set_xlim([700, 3000])
    subtitle_color = subtitle_colors[i]
    plot_color = plot_colors[i]
    ax.set_title(f'Model {i+1}', color=subtitle_color)
    drawModel1D(ax, thickness=thk, values=inv, plot='semilogx', color=plot_color, linewidth=3)

plt.tight_layout()
plt.show()



#%% Comparison of Synthetic Models, Geophilus sounding curves and Inversions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pygimli.viewer.mpl import drawModel1D

# Synthetic model Rho data
Rhos1 = np.array([800, 2500])
Rhos2 = np.array([2500, 800])
Rhos3 = np.array([2500, 800])
Rhos4 = np.array([800, 2500])
Rhos5 = np.array([800, 3000, 800])
Rhos6 = np.array([800, 3000, 800])

# Synthetic model thickness data
thk_syn1 = np.array([0.4])
thk_syn2 = np.array([0.4])
thk_syn3 = np.array([1.5])
thk_syn4 = np.array([1.5])
thk_syn5 = np.array([1, 0.3])
thk_syn6 = np.array([0.3, 0.2])

thk_syn_list = [thk_syn1, thk_syn2, thk_syn3, thk_syn4, thk_syn5, thk_syn6]
Rhos_list = [Rhos1, Rhos2, Rhos3, Rhos4, Rhos5, Rhos6]

colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# Create a multipage PDF
pdf_pages = PdfPages('comparison_Synthetic&inv-plots.pdf')

# Plot
fig, axs = plt.subplots(2, 6, figsize=(20, 8))
fig.suptitle('Synthetic Models, Geophilus sounding curves and Inversions', fontsize=14)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
extracted_rhos = []
extracted_rhos_withNoise = []

for i, (rho, thk_syn, color) in enumerate(zip([Rhos1, Rhos2, Rhos3, Rhos4, Rhos5, Rhos6], thk_syn_list, colors)):
    # sounding curves of synthetic models
    amVec = np.arange(1, 6) * 0.6
    b = 1.0
    bmVec = np.sqrt(amVec**2 + b**2)
    fop_syn = VESRhoModelling(thk_syn, am=amVec, an=bmVec, bm=bmVec, bn=amVec)

    # Generate synthetic sounding curve
    response_syn = fop_syn.response(rho)
    extracted_rhos.append(response_syn)     # extract Rhos from sounding curves 

    # Add random noise to the response_syn array
    noise_level = 0.05
    noise = np.random.randn(*response_syn.shape)
    response_syn_with_noise = response_syn * (noise +1)
    extracted_rhos_withNoise.append(response_syn_with_noise)  # extract Rhos from sounding curves with noise

    
    # inversion
    # Extract 5 Rhos from each synthetic sounding curves at b-values
    b_values = [0.6, 1.0, 1.5, 2.0, 2.5]

    # inversion result of synthetic models using extracted Rhos
    thk = np.ones(15) * 0.1
    fop = VESRhoModelling(thk, am=amVec, an=bmVec, bm=bmVec, bn=amVec)

    # List to store inversion results for each b value
    inversion_results = []
    inversion_results_Noise = []
    # List to store inversion results for each synthetic model
    Stmodels = []
    
    # inversion
    error = np.ones_like(response_syn) * 0.03
    inv_indiv = Inversion(fop=fop)
    inv_indiv.setRegularization(cType=1)
    modelInd = inv_indiv.run(response_syn, error, lam=350, zWeight=3, lambdaFactor=1, startModel=300, verbose=True,
                             blockyModel=True)
    inversion_results.append(modelInd)

    # Save inversion results for this synthetic model
    Stmodels.append(inversion_results)
          
    # inversion With Noise
    # error_noise = np.ones_like(response_syn_with_noise) * 0.03
    # inv_indiv = Inversion(fop=fop)
    # inv_indiv.setRegularization(cType=1)
    # modelInd_N = inv_indiv.run(response_syn_with_noise, error_noise, lam=350, zWeight=3, lambdaFactor=1.12, startModel=300, verbose=True)
    # inversion_results_Noise.append(modelInd_N)

    # Plot synthetic models
    drawModel1D(axs[0, i], thickness=thk_syn, values=rho, plot='semilogx', color=color, linewidth=3, zmax=2.5, label='synthetic')
    axs[0, i].set_ylim([2, 0])
    axs[0, i].set_xlim([500, 5000])
    axs[0, i].set_title(f'Model {i+1}', fontsize=12)
    axs[0, i].grid(True)
    
    # Plot the inversion result
    drawModel1D(axs[0, i], thickness=thk, values=modelInd, plot='semilogx', color='black', linewidth=3, zmax=2, label='inversion')

    # Plot the inversion result With Noise
    #drawModel1D(axs[0, i], thickness=thk, values=modelInd_N, plot='semilogx', color='y', linewidth=1.5, zmax=2, label='inversion')

    # Plot the sounding curve
    axs[1, i].semilogx(response_syn, amVec, label=f'Synthetic Model {i+1}', color=color, linewidth=3)
    axs[1, i].set_ylim([2.5, 0.6])
    axs[1, i].set_xlim([700, 3000])
    axs[1, i].set_title(f'Sounding Curves, Model {i+1}', fontsize=12)
    axs[1, i].set_xlabel('Apparent Resistivity $(\Omega$m$)$', fontsize=10)
    axs[1, i].set_ylabel('b (m)')
    axs[1, i].grid(True)
    axs[1, i].invert_yaxis()


# Save the current figure into the multipage PDF
pdf_pages.savefig(fig, bbox_inches='tight')
    
    
    
# Close the multipage PDF
pdf_pages.close()

# Save as JPEG
plt.savefig('comparison_Synthetis&inv-plots.jpg', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()


#%% Plot sounding curves one by one

import numpy as np
import matplotlib.pyplot as plt
import utm

def read_data(filepath):
    try:
        EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, nan_data, Rho5, Gamma, BFI, lfd, Datum, Zeit = np.genfromtxt(filepath, skip_header=1, delimiter=';', unpack=True)
        Eutm, Nutm, zone, letter = utm.from_latlon(NdecDeg, EdecDeg, 33, 'U')
        data = np.column_stack((Eutm, Nutm, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma))
        return data
    except ValueError:
        EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, nan_data, Rho5, Gamma, BFI, lfd, Datum, Zeit = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
        Eutm, Nutm, zone, letter = utm.from_latlon(NdecDeg, EdecDeg, 33, 'U')
        data = np.column_stack((Eutm, Nutm, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma))
        return data



def read_data_Aus(filepath):
    data_aus= np.genfromtxt(filepath, skip_header=1, delimiter=';', unpack=True)
    #achseFFT = data_aus[55]  
    # Find indices where Achse-5-U-FFT-Frq is not equal to 50
    indices_to_remove = np.where(data_aus[54] != 50)
    
    # Remove rows where Achse5-U-FFT-Frq is not equal to 50
    filtered_data = np.delete(data_aus, indices_to_remove, axis=1)
    
    EdecDeg, NdecDeg, H = filtered_data[0], filtered_data[1], filtered_data[2]
    Rho1, Rho2, Rho3, Rho4, Rho6 = filtered_data[11], filtered_data[12], filtered_data[13], filtered_data[14], filtered_data[16]
    Gamma = filtered_data[30]
    Eutm, Nutm, zone, letter = utm.from_latlon(NdecDeg, EdecDeg, 33, 'U')
    data = np.column_stack((Eutm, Nutm, H, Rho1, Rho2, Rho3, Rho4, Rho6, Gamma))
    return data


# Provide the path to your data file
filepath_Original = 'Aussenschlag_2023_10_11.csv'
farmName_Original = 'Trebbin Aussenschlag'


# Read the data
data = read_data_Aus(filepath_Original)

# Select one of the soundings (adjust the index accordingly)
selected_point_index = 6000
selected_point = data[selected_point_index]

# Plot the sounding curve
amVec = np.arange(1, 6) * 0.6
b = 1.0
bmVec = np.sqrt(amVec**2 + b**2)
#response_syn = VESRhoModelling(selected_point[3:8], am=amVec, an=bmVec, bm=bmVec, bn=amVec).response(selected_point[8])

plt.figure()
plt.semilogx(selected_point[3:8], amVec, label=f'Sounding Curve for Point {selected_point_index + 1}', linewidth=3)
plt.ylim([2.5, 0.6])
plt.xlim([10, 1000])
plt.title(f'Sounding Curve for Point {selected_point_index}', fontsize=12)
plt.xlabel('Resistivity $(\Omega$m$)$', fontsize=10)
plt.ylabel('b (m)')
plt.grid(True)
#plt.gca().invert_yaxis()
plt.legend()
axs[1, i].invert_yaxis()
# Save the figure before showing
plt.savefig(f'Sounding_Curve_Point_{selected_point_index}.jpg', dpi=300)

# Show the plot
plt.show()



#%% Sounding Curves for Selected Points

# Provide the path to your data file
filepath_Original = 'Aussenschlag_2023_10_11.csv'
farmName_Original = 'Trebbin Aussenschlag'
# Read the data
data = read_data_Aus(filepath_Original)

# Selected sounding indices
selected_indices = [124, 125, 126, 281, 2505, 1101, 555, 550, 1102, 4020]

# Plot the sounding curves together in a 2x5 subplot
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sounding Curves for Selected Points', fontsize=14)

for i, idx in enumerate(selected_indices):
    selected_point = data[idx]

    amVec = np.arange(1, 6) * 0.6
    b = 1.0
    bmVec = np.sqrt(amVec**2 + b**2)

    row, col = divmod(i, 5)
    axs[row, col].semilogx(selected_point[3:8], amVec, label=f'Point {idx + 1}')
    axs[row, col].set_ylim([2.5, 0.6])
    axs[row, col].set_xlim([10, 1000])
    axs[row, col].set_title(f'Point {idx + 1}', fontsize=10)
    axs[row, col].set_xlabel('Resistivity $(\Omega$m$)$', fontsize=8)
    axs[row, col].set_ylabel('b (m)')
    axs[row, col].grid(True)
    axs[row, col].invert_yaxis()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
plt.savefig('Combined_Sounding_Curves_Subplot.jpg', dpi=300)

# Show the plot
plt.show()
#%%

A = np.genfromtxt("Boossen_1601_1DInv_allLines_combined.txt", names=True, skip_header=0, delimiter=';')

fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
norm = LogNorm(10, 1000)
for i, a in enumerate(ax.flat):
    inv = A["inv{:02d}".format(i+1)]  # Adjust the field name here
    im = a.scatter(A["x"]-460000, A["y"]-5.8e6, c=inv, s=1, norm=norm, cmap="Spectral_r", alpha=1)
    a.set_aspect(1.0)
    a.set_title("z={:.1f}-{:.1f}m".format(i*.1, i*.1+.1))

plt.show()
