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
# %% import data

## Define file paths and farm names
# filepath = 'Geophilus_aufbereitet_2023-04-05_New.csv'
# farmName = 'aufbereitet'

# filepath = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName = 'Trebbin Wertheim'

filepath = 'Aussenschlag_2023_10_11.csv'
farmName_Original = 'Trebbin Aussenschlag'

# filepath = 'BFD_Banane_Geophilus_221118_roh.csv'
# farmName = 'Beerfelde Banane'

# filepath = 'BFD_Fettke_Geophilus_roh.csv'
# farmName = 'Beerfelde Fettke'

# filepath_Processed = 'BFD_Fettke_Geophilus_roh.csv'
# farmName_Processed = 'Beerfelde_Fettke_Processed data'
# filepath_original = 'Beerfelde 2022_10_21-original.csv'
# farmName_original = 'Beerfelde_Fettke_Original data'


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
from Data_Processing import read_data_Aus_lineIndexed
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

filepath_Original = 'Aussenschlag_2023_10_11.csv'
farmName_Original = 'Trebbin Aussenschlag'
data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

farm_name = 'Trebbin Aussenschlag'
refpoints = f'{farmName_Original} farm_reference_Points.csv'
RefPoints = pd.read_csv(refpoints, delimiter=";")

line_number = 21
amVec = np.arange(1, 6) * 0.6  
#selected_line_data = extract_data_from_selected_line(data_lineIndexed_file, line_number=line_number, amVec=amVec)

Stmodels, chi2 = inversion_1D_line(selected_line_data, amVec, farmName_Original, line_number)




#
#%% plot 1D inversion result for one line (fn)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

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
        for i, (e, n, label) in enumerate(zip(RefPoints['E'], RefPoints['N'], RefPoints['Name'])):
            ax3.plot(e, n, "x", markersize=8, mew=2)
            ax3.text(e, n, label, ha='right', va='bottom', fontsize=8)
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
# filepath_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName_Processed = 'Trebbin Wertheim_Processed'
# filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName = 'Boossen_1601'
# # Read harmfit data
# harmfitted_data_file = f"harmfit_on_original_{farmName_Original}.txt"

# #harmfit = f'harmfit_on_original_{farmName}.txt'
# #data_file =harmfit

# smoothed_harmfitted_data_file = f"smoothed_harmfit_{farmName_Original}.txt"
# smoothedharmfit_data = read_data_UTM(smoothed_harmfitted_data_file)

filepath_Original = 'Aussenschlag_2023_10_11.csv'
farmName_Original = 'Trebbin Aussenschlag'
data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

line_number = 21

chi2 = pd.read_csv(f'{farmName_Original}_chi2_indiv-1dInv_line_number_{line_number}.csv', delimiter=';', header=None)
inv_result = np.loadtxt(f'{farmName_Original}_1DInv_line_number_{line_number}.txt', delimiter=';')
refpoints = f'{farmName_Original} farm_reference_Points.csv'
RefPoints = pd.read_csv(refpoints, delimiter=";")

# # selected_line_data = extract_data_from_selected_line(data_file, max_angle_deviation, line_number=line_number, amVec=np.arange(1, 6) * 0.6)
thk = np.ones(15) * 0.1
#selected_line_data = extract_data_from_selected_line(data_lineIndexed_file, line_number=line_number, amVec=amVec)

plot_1d_inversion_results(filepath_Original, line_number, selected_line_data, RefPoints, thk, chi2, inv_result)


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

#%% 1D inversion for all lines
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

    data = read_data_Aus_lineIndexed(filepath)
    data_filtered = data[data[:, -1] != 0] # remove individual points
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
    for line_idx, line_num in enumerate(selected_lines, start=1):
        point_indices_of_interest = np.where(data_filtered[:,-1] == line_num)[0]
        data_for_inv = data_filtered[:, 3:8][point_indices_of_interest]
        error = np.ones_like(data_for_inv[1]) * 0.03
        
        for j, spacing in enumerate(spacing_labels):
            chi2Vec_indiv = []
            Stmodels = []
            results_list = []
            inv_response_all_soundings = []

            # Create an error matrix for individual data
            error = np.ones_like(data_for_inv[j]) * 0.03
            for i, indivData in enumerate(data_for_inv):
                inv_indiv = Inversion(fop=fop)  # Passing the fwd operator in the inversion
                inv_indiv.setRegularization(cType=1)  # cType=1 for smoothness constraint
                modelInd = inv_indiv.run(indivData, error, lam=350, zWeight=3, lambdaFactor=1.12, startModel=300, verbose=True)
                Stmodels.append(modelInd)

                xy_coord = data_for_inv[i, 0:2]  # Extract x and y coordinates
                chi2 = inv_indiv.inv.chi2()
                chi2Vec_indiv.append(chi2)

                # Append the results for the current point
                results_list.append(np.hstack((xy_coord, modelInd, chi2)))
                
                # forward model response
                inv_response = inv_indiv.response
                inv_response_all_soundings.append(inv_response)

            # Save results for the current line
            model_filename = f'{farmName}_1D-InvResult_allLines_line_number_{line_idx}.txt'
            np.savetxt(model_filename, np.array(Stmodels), delimiter='\t')

            chi2_filename =  f'{farmName}_chi2_1DInv_allLines_line_number_{line_idx}.csv'
            np.savetxt(chi2_filename, np.array(chi2Vec_indiv), delimiter=',')
            
            result_file = f'{farmName}_1Dinv_response_all_lines_line_number_{line_idx}.txt'
            # Save the data with the header
            header = "x; y; " + "; ".join([f"inv{k + 1}" for k in range(len(modelInd))]) + "; chi2"
            np.savetxt(result_file, results_list, delimiter=';', fmt='%f', header=header)
            
            inv_response_filename = f'{farmName}_1Dinv_response_line_number_{line_idx}.txt'
            np.savetxt(inv_response_filename, np.array(inv_response_all_soundings), delimiter=';')

        chi2_indiv_list.append(chi2Vec_indiv)
        Stmodels_list.append(Stmodels)

    return Stmodels_list, chi2_indiv_list


# Example usage:
# data_file = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName = 'Trebbin Wertheim'

# filepath_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName_Processed = 'Trebbin Wertheim_Processed'

# filepath_Processed = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1601'
# filepath_Processed = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName_Processed = 'Boossen_1211'
filepath_Original = 'Aussenschlag_2023_10_11.csv'
farmName_Original = 'Trebbin Aussenschlag'
data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
Stmodels, chi2 , Stmodels_with_columns = inversion_1D_all_lines(data_lineIndexed_file, amVec, farmName_Original)

# inv for processed
farmName_Processed = 'Trebbin Aussenschlag_Processed'
filepath_Processed = 'Aussenschlag_2023-10-11_aufbereitet.csv'
data_lineIndexed_file_p = f"data_lineIndexed_{farmName_Processed}.txt"
Stmodels, chi2  = inversion_1D_all_lines(data_lineIndexed_file_p, amVec, farmName_Processed)

#%% 1D inversion plotting for all lines 

def plot_1d_inversion_results_allLines(filepath, amVec, farmName):
    data = read_data_Aus_lineIndexed(filepath)
    data_filtered = data[data[:, -1] != 0] # remove individual points
    selected_lines = np.unique(data_filtered[:,-1])
    line_data = {}  # Dictionary to hold data for each line

    with PdfPages(f'{farmName}_1D_Inversion_all_lines.pdf') as pdf:
        for line_num in selected_lines:
            line_mask = (data_filtered[:, -1] == line_num)
            line_data[line_num] = data_filtered[line_mask][:, 3:8]
            mydata = np.asarray(line_data[line_num])
            Inv_indiv = np.loadtxt(f'{farmName}_1D-InvResult_allLines_line_number_{int(line_num)}.txt', delimiter='\t')
            chi2 = pd.read_csv(f'{farmName}_chi2_1DInv_allLines_line_number_{int(line_num)}.csv')
            chi2Indiv = np.array(chi2.T)
            #mydata = data_filtered[:, 3:8]
            # Eutm, Nutm = data_filtered[:, 0], data_filtered[:, 1]
            Eutm, Nutm = data_filtered[line_mask][:, 0] , data_filtered[line_mask][:, 1]
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15.5, 8), gridspec_kw={'height_ratios': [1, 1]})
    
            fig.suptitle(f'1D_Inversion (line_{int(line_num)}, Farm Name: {farmName_Original})', fontsize=16)
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
            inv_response = np.loadtxt(f'{farmName}_1Dinv_response_line_number_{int(line_num)}.txt', delimiter=';')
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
            ax3.plot(Eutm, Nutm, "o", markersize=4, mew=1, label=f'line {int(line_num)}')
            # Plot the reference points and add labels
            for i, (e, n, label) in enumerate(zip(RefPoints['E'], RefPoints['N'], RefPoints['Name'])):
                ax3.plot(e, n, "x", markersize=8, mew=2)
                ax3.text(e, n, label, ha='right', va='bottom', fontsize=8)
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
            x = np.arange(mydata.shape[0])
            y = np.arange(mydata.shape[1])
            y_ticks = y[::-1]
            extent = [x[0], x[-1], y[0], y[-1]]
            # Display the numerical values with the color map
            #im = ax4.imshow(mydata, cmap='Spectral_r', norm=norm, extent=extent)
            im = ax4.imshow(mydata.T, cmap='Spectral_r', norm=norm, extent=extent, interpolation="nearest")
            ax4.set_aspect('auto')  
            ax4.set_title('Rhoa')
            ax4.set_yticks(y)
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

filepath_Original = 'Aussenschlag_2023_10_11.csv'
farmName_Original = 'Trebbin Aussenschlag'
data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"
    
# pdf_filename = f'{farmName_Processed}_InversionResults.pdf'
plot_1d_inversion_results_allLines(data_lineIndexed_file, amVec, farmName_Original)

# inv plot for processed
farmName_Processed = 'Trebbin Aussenschlag_Processed'
filepath_Processed = 'Aussenschlag_2023-10-11_aufbereitet.csv'
data_lineIndexed_file_p = f"data_lineIndexed_{farmName_Processed}.txt"
plot_1d_inversion_results_allLines(data_lineIndexed_file_p, amVec, farmName_Processed)
