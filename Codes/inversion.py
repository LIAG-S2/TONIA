#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:43:24 2023

@author: sadegh
"""
# %% import lberaries
import utm
import fiona
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from utm import to_latlon
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from pygimli.frameworks import harmfit
from Data_Processing import DataProcessing
from sklearn.ensemble import IsolationForest
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

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
        
# %% inversion class
class InversionClass:
    def __init__(self, data_processor):
        self.data_processor = data_processor

        pass
        
    #% 1D Inversion for all lines
    def inversion_1D_all_lines(self, filepath, farmName, Inv_File_name, survey_date):
        """
        Perform 1D inversion for all selected lines' data.
    
        Parameters:
        - selected_lines_data: List of numpy arrays containing data for each selected line.
        - amVec: 
        - farmName: Name of the farm or site for saving results.
    
        Returns:
        - chi2_indiv_list: List of lists of chi-square values for individual inversions of each line.
        - Stmodels_list: List of lists of inversion models for each line.
        """
        # Get the current date in the format YY-MM-DD
        current_date = datetime.now().strftime('%y-%m-%d')
    
        spacing_labels = ['S1', 'S2', 'S3', 'S4', 'S5']
    
        data = self.read_data_lineIndexed(filepath)
        data_no_nan = data[~np.isnan(data).any(axis=1)]
        data_filtered = data_no_nan[data_no_nan[:, -1] != 0] # remove individual points
        selected_lines = np.unique(data_filtered[:,-1])
    
        # Initialize ABMN values based on amVec
        c = 0.6  # v1/v2, 0.6 for v3
        a = 1.0
        nMN = 5  # 6 for Geophilus v2
        b = np.arange(1, nMN+1) * 0.6  # Geophilius 3.0 (2020)
        bmVec = np.sqrt(b**2+a**2)
        k = np.pi * b / (1-b/bmVec)
    
        # Model space
        thk = np.ones(19) * 0.1
        # Initialize the DC 1D Forward Modelling Operator
        fop = VESRhoModelling(thk, am=b, an=bmVec, bm=bmVec, bn=b)
        chi2_indiv_list = []
        Stmodels_list = []
        invResult_AllLines = []
        forward_resp_AllLines = []
        for line_idx, line_num in enumerate(selected_lines, start=1):
            point_indices_of_interest = np.where(data_filtered[:,-1] == line_num)[0]
            data_for_inv = data_filtered[point_indices_of_interest]
            
            chi2Vec_indiv = []
            Stmodels = []
            results_list = []
            inv_response_all_soundings = []
            xy_coord_currentLine = []
            xy_coord_LatLong = []
            h_currentLine = []
            gamma_currentLine = []
            
            # Create an error matrix for individual data
            rhoa = data_for_inv[:, 3:8]
            I =  0.1  # 100mA
            U = rhoa * I / k
            dU = 0.01  # print(dU/U)
            dR = dU / I
            error = dR * k / rhoa + 0.02
            for i, indivData in enumerate(data_for_inv):
                inv_indiv = Inversion(fop=fop)  # Passing the fwd operator in the inversion
                inv_indiv.setRegularization(cType=1)  # cType=1 for smoothness constraint
                # modelInd = inv_indiv.run(indivData[3:8], error, lam=350, zWeight=3, lambdaFactor=1.12, startModel=300, verbose=True)
                lam = 30
                modelInd = inv_indiv.run(indivData[3:8], error[i], lam=lam, startModel=300, verbose=True)
                Stmodels.append(modelInd)
    
                xy_coord = data_for_inv[i, 0:2]  # Extract x and y coordinates
                xy_coord_currentLine.append(xy_coord)
                h = data_for_inv[i, 2]
                h_currentLine.append(h)
                gamma = data_for_inv[i, -2]
                gamma_currentLine.append(gamma)
                # chi2 = inv_indiv.inv.chi2()
                chi2 = inv_indiv.chi2() 
                chi2Vec_indiv.append(chi2)
                line_index = np.array(data_for_inv[:,-1])
                
                # Append the results for the current point
                results_list.append(np.hstack((xy_coord, h, modelInd, chi2, gamma)))
                
                # forward model response
                inv_response = inv_indiv.response
                inv_response_all_soundings.append(inv_response)
                
            # convert the UTM coordinates back to latitude and longitude
            for coords in xy_coord_currentLine:
                Eutm, Nutm = coords
                latitude, longitude = to_latlon(Eutm, Nutm, 33, 'U')
                xy_coord_LatLong.append((longitude, latitude))
    
          
            # making an array for the inversion result for all data
            invResult_currentLine = np.hstack((np.array(xy_coord_LatLong), 
                                               np.reshape(h_currentLine, (-1, 1)), 
                                               np.array(Stmodels), 
                                               np.reshape(chi2Vec_indiv, (-1, 1)), 
                                               np.reshape(gamma_currentLine, (-1, 1)), 
                                               np.reshape(line_index, (-1, 1))))
            
            invResult_AllLines.append(invResult_currentLine)
            invResult_combined = np.concatenate(invResult_AllLines, axis=0)
            header = "X; Y; H;" + "; ".join([f"ERi{k + 1}" for k in range(len(modelInd))]) + "; Chi2;  Gamma; line_index"
            
            # save the inv result with 2 different names
            result_file_1 = f'1Dinv_response_all_lines_{farmName}.csv'
            np.savetxt(result_file_1, invResult_combined, delimiter=';', fmt='%.8f', header=header)
            result_file_2_with_date = f'{Inv_File_name}_{current_date}_sd{survey_date}.csv'
            np.savetxt(result_file_2_with_date, invResult_combined, delimiter=';', fmt='%.8f', header=header, comments='')
    
            # making an array for the forward response for all data    
            forward_resp_currentLine = np.hstack((np.array(xy_coord_LatLong), np.array(inv_response_all_soundings), 
                                               np.reshape(line_index, (-1, 1))))
            forward_resp_AllLines.append(forward_resp_currentLine)
            forward_resp_combined = np.concatenate(forward_resp_AllLines, axis=0)
            header_forward = "X; Y; " + "; ".join([f"ForwResp{k + 1}" for k in range(5)]) + ";  line_index"
            result_file_forward = f'1Dforward_response_all_lines_{farmName}.txt'
            np.savetxt(result_file_forward, forward_resp_combined, delimiter='\t', fmt='%.8f', header=header_forward)
            
            chi2_indiv_list.append(chi2Vec_indiv)
            Stmodels_list.append(Stmodels)
    
        return Stmodels_list, chi2_indiv_list
    
    
    def read_data_lineIndexed(self, filepath):
        EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma, line_index = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
        # EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, nan_data, Rho5, Gamma, BFI, lfd, Datum, Zeit = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
        data = np.column_stack((EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma, line_index))
        return data
    
    #% 1D inversion plotting for all lines 
    def plot_1d_inversion_results_allLines(self, filepath, farmName, data_type, thk, kmlFile):
        data = self.read_data_lineIndexed(filepath)
        data_no_nan = data[~np.isnan(data).any(axis=1)]
        data_filtered = data_no_nan[data_no_nan[:, -1] != 0] # remove individual points
        selected_lines = np.unique(data_filtered[:,-1])
        line_data = {}  # Dictionary to hold data for each line
        lam = 30
        # Import reference points
        refpoints = self.import_reference_points_from_kml(kmlFile, filepath)
        ref_Name = refpoints[:, 0].astype(float)
        ref_eutm = refpoints[:, 1].astype(float)
        ref_nutm = refpoints[:, 2].astype(float)


        with PdfPages(f'1D_Inversion_all_lines_{farmName}.pdf') as pdf:
            for line_num in selected_lines:
                line_mask = (data_filtered[:, -1] == line_num)
                line_data[line_num] = data_filtered[line_mask][:, 3:8]
                mydata = np.asarray(line_data[line_num])
                
                # Load the inversion results from the saved file
                inv_result_data = np.loadtxt(f'1Dinv_response_all_lines_{farmName}.csv', delimiter=';', skiprows=1)
                # Extract relevant information
                x_values = inv_result_data[:, 0]
                y_values = inv_result_data[:, 1]
                inversion_results = inv_result_data[:, 3:-3]  # Exclude x, y, chi2, and line_index columns
                chi2_values = inv_result_data[:, -2]
                line_index_values = inv_result_data[:, -1]
    
                # extract inv result for the current line
                inv_result_data_current_line = inv_result_data[inv_result_data[:, -1] == line_num]
                Inv_indiv = inv_result_data_current_line[:, 3:-3]
                chi2 = inv_result_data_current_line[:, -3]
                chi2Indiv = np.array(chi2.T)
    
                Eutm, Nutm = data_filtered[line_mask][:, 0] , data_filtered[line_mask][:, 1]
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15.5, 8), gridspec_kw={'height_ratios': [1, 1]})
        
                fig.suptitle(f'1D_Inversion (Farm Name: {farmName} , line_{int(line_num)})', fontsize=16)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
                ax0, ax1, ax2, ax3, ax4, ax5 = axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]
                
                
                ax0.set_xlim([min(Eutm), max(Eutm)])
                ax0.set_ylim([min(Nutm), max(Nutm)])
                ax0.axis("equal")
                ax0.plot(Eutm, Nutm, "o", markersize=4, mew=1, label=f'line {int(line_num)}')
                # Plot reference points along with the lines
                for name, x, y in zip(ref_Name.astype(int), ref_eutm, ref_nutm):
                    ax0.scatter(x, y, color='red')
                    ax0.annotate(str(name), (x,y), textcoords="offset points", xytext=(0, 5), ha='center')

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
                forward_resp = np.loadtxt(f'1Dforward_response_all_lines_{farmName}.txt', delimiter='\t', skiprows=1)
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
    

    
    def subplot_inversion_results(self, filepath, farm_name):
        zone = 33
        hemisphere = 'U'
        
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filepath, delimiter=';', skip_header=1)
        
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        Eutm, Nutm, zone_, letter_ = utm.from_latlon(y_values_Long, x_values_Lat, zone, hemisphere)
        
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
        
        # Calculate the dynamic figsize based on the range of Easting and Northing
        x_range = np.ptp(Eutm)  # Peak-to-peak (max - min) range of Easting coordinates
        y_range = np.ptp(Nutm)  # Peak-to-peak (max - min) range of Northing coordinates
        
        # Scaling factors to determine the figure size
        x_scale = 0.01  # Adjust this scaling factor to control width
        y_scale = 0.01  # Adjust this scaling factor to control height
        
        # Calculate the figure size dynamically
        fig_width = max(8, x_range * x_scale)  # Set a minimum width to avoid too small plots
        fig_height = max(6, y_range * y_scale)  # Set a minimum height to avoid too small plots
        
        pdf_filename = f'Inversion_Results_{farm_name}.pdf'
        with PdfPages(pdf_filename) as pdf:
            fig, ax = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True, figsize=(fig_width, fig_height))
            norm = LogNorm(10, 1000)
            for i, a in enumerate(ax.flat):
                if i + 3 < inv_res.shape[1]:  # Ensure we don't go out of bounds
                    rho = inv_res[:, i+3]  # Assuming the rho values start from the third column
                    im = a.scatter(Eutm-x_offset, Nutm-y_offset, c=rho, s=0.1, norm=norm, cmap="Spectral_r", alpha=1)
                    a.set_aspect(1.0)
                    a.set_title("z={:.1f}-{:.1f}m".format(i*0.1, (i+1)*0.1))
                else:
                    a.axis('off')  # Turn off the axis if there's no data to display
        
            # Add a color bar
            cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Resistivity (Ohm.m)')
        
            # Set common labels
            fig.text(0.5, 0.02, 'Easting', ha='center', va='center')
            fig.text(0.02, 0.5, 'Northing', ha='center', va='center', rotation='vertical')
        
            plt.suptitle(f'{farm_name} Inversion Results', fontsize=16)
            plt.tight_layout(rect=[0.03, 0.03, 0.93, 0.95])  # Adjust the layout to make room for the suptitle
            pdf.savefig(fig)
        
            plt.show()

    
    def subplot_inversion_limit_chi2(self, filepath, farm_name, Inv_File_name, chi2_limit=20, zone=33, hemisphere='U'):
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filepath, delimiter=';', skip_header=1)
        # Filter out rows where the Chi2 is greater than the limit
        inv_res_filtered = inv_res[inv_res[:, -3] <= chi2_limit]
    
        # Save the filtered inversion results to a text file
        header_inv = "X; Y; H;" + "; ".join([f"ERi{i}" for i in range(1, inv_res_filtered.shape[1]-5)]) + "; Chi2;  Gamma; line_index"
    
        current_date = datetime.now().strftime('%y-%m-%d')
        np.savetxt(f'{farm_name}_1Dinv_response_all_lines_combined_chi2_less_than_{chi2_limit}.txt', 
                   inv_res_filtered, delimiter=';', header=header_inv, fmt='%.8f', comments='')
        np.savetxt(f'{Inv_File_name}_{current_date}_chi2_lt_{chi2_limit}.csv', 
                   inv_res_filtered, delimiter=';', header=header_inv, fmt='%.8f', comments='')
    
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res_filtered[:, 0], inv_res_filtered[:, 1]
        Eutm, Nutm, zone_, letter_ = utm.from_latlon(y_values_Long, x_values_Lat, zone, hemisphere)
    
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
    
        # Calculate the dynamic figsize based on the range of Easting and Northing
        x_range = np.ptp(Eutm)  # Peak-to-peak (max - min) range of Easting coordinates
        y_range = np.ptp(Nutm)  # Peak-to-peak (max - min) range of Northing coordinates
    
        # Scaling factors to determine the figure size
        x_scale = 0.01  # Adjust this scaling factor to control width
        y_scale = 0.01  # Adjust this scaling factor to control height
    
        # Calculate the figure size dynamically
        fig_width = max(8, x_range * x_scale)  # Set a minimum width to avoid too small plots
        fig_height = max(6, y_range * y_scale)  # Set a minimum height to avoid too small plots
    
        pdf_filename = f'Inversion_Results_{farm_name}_chi2_lt_{chi2_limit}.pdf'
        with PdfPages(pdf_filename) as pdf:
    
            fig, ax = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True, figsize=(fig_width, fig_height), 
                                   gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    
            norm = LogNorm(10, 1000)
            for i, a in enumerate(ax.flat):
                if i + 3 < inv_res_filtered.shape[1]:  # Ensure we don't go out of bounds
                    rho = inv_res_filtered[:, i+3]  # Assuming the rho values start from the third column
                    im = a.scatter(Eutm-x_offset, Nutm-y_offset, c=rho, s=0.1, norm=norm, cmap="Spectral_r", alpha=1)
                    a.set_aspect(1.0)
                    a.set_title("z={:.1f}-{:.1f}m".format(i*0.1, (i+1)*0.1))
                else:
                    a.axis('off')  # Turn off the axis if there's no data to display
            plt.tight_layout(rect=[0.03, 0.03, 0.93, 0.95])  # Adjust the layout to make room for the suptitle
    
            # Add a color bar
            cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Resistivity (Ohm.m)')
    
            # Set common labels
            fig.text(0.5, 0.02, 'Easting', ha='center', va='center')
            fig.text(0.05, 0.5, 'Northing', ha='center', va='center', rotation='vertical')
    
            plt.suptitle(f'{farm_name} Inversion Results, chi2<{chi2_limit}', fontsize=16)
            pdf.savefig(fig)
    
            plt.show()
  

    
    #% inversion result multipage pdf
    def plot_inv_results_multipage_pdf(self, filename, farm_name, zone=33, hemisphere='U'):
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filename, delimiter=';', skip_header=1)
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        Eutm, Nutm, zone_, letter_ = utm.from_latlon(y_values_Long, x_values_Lat, zone, hemisphere)

        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
    
        # Create a multi-page PDF file
        pdf_filename = f'inversion_results_{farm_name}_MPpdf.pdf'
        with PdfPages(pdf_filename) as pdf:
            for i in range(20):  # Loop through all subplots
                fig, ax = plt.subplots(figsize=(8, 6))
                norm = LogNorm(10, 1000)
                rho = inv_res[:, i + 3]  # Assuming the rho values start from the third column
                im = ax.scatter(Eutm-x_offset, Nutm-y_offset, c=rho, s=0.8, norm=norm, cmap="Spectral_r", alpha=1)
                ax.set_title("z={:.1f}-{:.1f}m".format(i * 0.1, (i + 1) * 0.1))
                # Add color bar
                cbar = plt.colorbar(im, pad=0.02, shrink=0.8)
                cbar.set_label('Resistivity (Ohm.m)')
                # Set common labels
                ax.set_xlabel('Easting')
                ax.set_ylabel('Northing')
                ax.axis("equal")

                plt.suptitle(f'Inversion Results for {farm_name}', fontsize=16)
    
                # Save the subplot into the PDF file
                pdf.savefig(fig)
                plt.show()
                plt.close(fig)  # Close the figure to release memory
                

    #% plot Chi2
    def plot_chi2_results(self, filename, farm_name, zone=33, hemisphere='U'):
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filename, delimiter=';', skip_header=1)
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        Eutm, Nutm, zone_, letter_ = utm.from_latlon(y_values_Long, x_values_Lat, zone, hemisphere)
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
    
        # Create a multi-page PDF file
        pdf_filename = f'{farm_name}_Chi2_inversion_results.pdf'
        with PdfPages(pdf_filename) as pdf:
            fig, ax = plt.subplots(figsize=(8, 6))
            norm = LogNorm(1, 300)
            chi2 = inv_res[:, -3]  # Assuming the rho values start from the third column
            im = ax.scatter(Eutm-x_offset, Nutm-y_offset, c=chi2, s=0.2, norm=norm, cmap="Spectral_r", alpha=1)
            ax.set_aspect(1.0)
            ax.set_title(r"$\chi^2$")
            # Add color bar
            cbar = plt.colorbar(im, pad=0.02, shrink=0.5)
            cbar.set_label(r"$\chi^2$")
            # Set common labels
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            plt.suptitle(f'{farm_name}', fontsize=16)

            # Save the subplot into the PDF file
            plt.show()
    
            pdf.savefig(fig)
            plt.close(fig)


    def import_reference_points_from_kml(self, kml_file, farmName):
        # Import KML of reference points
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
        gdf = gpd.read_file(kml_file)
        
        # Extract latitude and longitude coordinates
        ref_lat = np.array(gdf['geometry'].y)
        ref_lon = np.array(gdf['geometry'].x)
    
        # Convert latitude and longitude to UTM
        ref_eutm, ref_nutm, ref_zone, ref_letter = utm.from_latlon(ref_lat, ref_lon)

        # Extract reference point names
        ref_name = np.array(gdf['Name'])
    
        # Create a table with columns: 'Name', 'E', 'N'
        refpoints = np.column_stack([ref_name, ref_eutm, ref_nutm])
    
        # Define the header
        header = ['Name', 'E', 'N']
    
        # Stack the header on top of the reference points
        ref_points_table = np.vstack((header, refpoints))
    
        # Save the reference points to a CSV file
        csv_filename = f'{farmName}_farm_reference_Points.csv'
        np.savetxt(csv_filename, ref_points_table, delimiter=';', fmt='%s')
        print(f'Reference points saved to {csv_filename}')
        return refpoints


#%% Plot inversion for nearest point to ref points
    def plot_inv_column_close_to_refs(self, filepath, farmName, kmlFile):
        """
        Plot inversion results for points close to reference points.
    
        Args:
            filepath (str): Path to the data file.
            farmName (str): Name of the farm.
            kmlFile (str): Path to the KML file.
    
        Returns:
            list: List of inversion models.
        """
        #data = self.read_data_lineIndexed(filepath)

        # Find nearest points to the reference points
        nearest_points_list, refpoints = self.data_processor.find_nearest_points_to_reference_for_processed_data(filepath, farmName, kmlFile)
    
        # Initialize ABMN values based on amVec
        c = 0.6  # v1/v2, 0.6 for v3
        a = 1.0
        nMN = 5  # 6 for Geophilus v2
        b = np.arange(1, nMN + 1) * 0.6  # Geophilius 3.0 (2020)
        bmVec = np.sqrt(b ** 2 + a ** 2)
        k = np.pi * b / (1 - b / bmVec)
    
        # Model space
        thk = np.ones(14) * 0.1  # =14 since we have the soil info only up to 1.5 m
        # data space
        thk_d = np.ones(5) * 0.1

        # Initialize the DC 1D Forward Modelling Operator
        fop = VESRhoModelling(thk, am=b, an=bmVec, bm=bmVec, bn=bmVec)
    
        # List to store inversion models
        invResult_nearest_points_combined = []
    
        for j, nearest_idx in enumerate(nearest_points_list):
            # Initialize lists to store results for each nearest point
            Stmodels = []
            xy_coord_currentLine = []
            h_currentLine = []
            rhoa_currentLine = []
    
            data_for_inv = nearest_idx
            rhoa = data_for_inv[:, 3:8]
            I = 0.1  # 100mA
            U = rhoa * I / k
            dU = 0.01
            dR = dU / I
            error = dR * k / rhoa + 0.02
    
            for i, indivData in enumerate(data_for_inv):
                # Run inversion for individual data
                inv_indiv = Inversion(fop=fop)
                inv_indiv.setRegularization(cType=1)
                lam = 30
                modelInd = inv_indiv.run(indivData[3:8], error[i], lam=lam, startModel=300, verbose=True)
    
                # Add nearest reference point name to the inversion model
                nearest_ref_name = refpoints[j, 0]
                modelInd_with_ref = np.column_stack((np.array(modelInd).reshape(1, -1), np.array(nearest_ref_name).reshape(-1, 1)))
                Stmodels.append(modelInd_with_ref)
    
                # Store other data for each point
                xy_coord = data_for_inv[i, 0:2]
                xy_coord_currentLine.append(xy_coord)
                h = data_for_inv[i, 2]
                h_currentLine.append(h)
                rhoa_currentLine.append(rhoa[i])  # Append rhoa for the current point
    
            # Convert UTM coordinates back to latitude and longitude
            xy_coord_LatLong = [(to_latlon(coords[0], coords[1], 33, 'U')) for coords in xy_coord_currentLine]
    
            # Combine data for each nearest point
            Stmodels_arr = np.array(Stmodels)
            Stmodels_reshaped = np.reshape(Stmodels_arr, (Stmodels_arr.shape[0], Stmodels_arr.shape[2]))
            invResult_nearest_points = np.hstack((np.array(xy_coord_LatLong),
                                                   np.reshape(h_currentLine, (-1, 1)),
                                                   np.array(Stmodels_reshaped)))
            invResult_nearest_points_combined.append(invResult_nearest_points)
    
        # Concatenate all arrays in invResult_nearest_points_combined along the row axis
        invResult_combined_all = np.concatenate(invResult_nearest_points_combined, axis=0)
        invResult_combined_all_float = invResult_combined_all.astype(float)
    
        # Define the header for the CSV file
        header = "X; Y; H;" + "; ".join([f"ERi{k + 1}" for k in range(len(modelInd))]) + "; Nearest Reference Point"
    
        # Save the concatenated array to a CSV file
        result_file = f'1Dinv_response_Nearest_{farmName}.csv'
        np.savetxt(result_file, invResult_combined_all_float, delimiter=';', fmt='%.8f', header=header)
    
    
        # Create subplots for each reference point to collect inv_result_rho
        inv_result_rho = []
        # Iterate over each reference point
        for idx1 in range(len(refpoints)):
            # Extract the reference point name
            ref_point_name = int(refpoints[idx1, 0])
            # Find the subset of inversion results corresponding to the reference point
            inv_result_subset = invResult_combined_all_float[np.where(invResult_combined_all_float[:, -1] == ref_point_name)]
    
            # Extract the model information (excluding the x, y, h coordinates and the reference point)
            models = inv_result_subset[:, 3:-1]
    
            # Append the model to the list of models
            inv_result_rho.append(models)
    
        # Create a PDF file to save the plots
        # Define the aspect ratio based on the number of columns and rows
        num_rows, num_cols = 2, 10
        aspect_ratio = (num_cols * 5) / (num_rows * 20)  # width / height
        
        # Define the height ratios for the rows
        height_ratios = [1, 0.5]  
        
        # Calculate the figure size based on the aspect ratio
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(40, 20), gridspec_kw={'height_ratios': height_ratios})

        fig.suptitle(f'1D Inversion result and Rhoa (processed) for the nearest points to the reference points (Farm Name: {farmName})', fontsize=28)
        plt.subplots_adjust(hspace=0.03)

        pdf_filename = f"1D_Inversion_Nearest_to_references_{farmName}.pdf"
        pdf_pages = PdfPages(pdf_filename)
    
        # Iterate over each reference point, inversion result and rhoa(org) for subplot
        aspect_ratio = float(inv_result_rho[0].shape[1]) / float(inv_result_rho[0].shape[0])
        from itertools import zip_longest
        for idx, (inv, ref_point, rhoa_p) in enumerate(zip_longest(inv_result_rho, refpoints, nearest_points_list, fillvalue=None)):
            # Extract the reference point name
            if ref_point is not None:
                ref_point_name = int(ref_point[0])
        
            # Plot stitched models for ER and rhoa in the first row
            norm = colors.LogNorm(vmin=10, vmax=1000)
            if inv is not None:
                axes[0, idx].imshow(inv.T, cmap='Spectral_r', norm=norm, interpolation="nearest")
                axes[0, idx].set_title(f'Inv close to ref {ref_point_name}')  # Set title for the first row
                
                # Set y-axis ticks and labels at the bottom of each level
                y_ticks = np.arange(0, len(thk) + 1)
                y_tick_labels = np.round(np.arange(0.1, 1.6, 0.1), 1)
                axes[0, idx].set_yticks(y_ticks + 0.5)  # Add a small offset to position ticks at the bottom
                axes[0, idx].set_yticklabels(y_tick_labels, rotation=0, ha='right', va='bottom')
                axes[0, idx].tick_params(axis='y', which='major')
            
                # Set x-axis ticks and labels
                x_ticks = np.concatenate((np.arange(-3, 0, 1), np.arange(1, 4, 1)))
                x_tick_labels = x_ticks
                axes[0, idx].set_xticks(np.arange(len(x_ticks)))
                axes[0, idx].set_xticklabels(x_tick_labels)

            # Plot stitched models for rhoa in the second row
            if rhoa_p is not None:
                # axes[1, idx].imshow((rhoa_p[:,3:-1]).T, cmap='Spectral_r', norm=norm, interpolation="nearest")
                # axes[1, idx].set_title(f'Rhoa close to ref {ref_point_name}')  # Set title for the first row
                x = np.arange(rhoa_p.shape[0])
                y = np.arange(rhoa_p.shape[1])
                #y_ticks = y[::-1]
                #extent = [x[0], x[-1], y[0], y[-1]]
                
                im = axes[1, idx].imshow((rhoa_p[:,3:-2]).T, cmap='Spectral_r', norm=norm, interpolation="nearest")
                axes[1, idx].set_aspect('auto')  
                axes[1, idx].set_title(f'Rhoa(proc) Ref Point {ref_point_name}')
                axes[1, idx].set_yticks(np.arange(0, len(thk_d)))  
                axes[1, idx].set_yticklabels(np.arange(1, 6, 1)) 
                axes[1, idx].set_ylabel('spacing')
                # Concatenate the negative and positive x-ticks
                x_ticks = np.concatenate((np.arange(-3, 0, 1), np.arange(1, 4, 1)))
                # Set the x-ticks and labels
                axes[1, idx].set_xticks(np.arange(len(x_ticks)))
                axes[1, idx].set_xticklabels(x_ticks)

        
        
        # Add color bar to the second row
        cbar = fig.colorbar(im, ax=axes[0, :], orientation='horizontal', shrink=0.3)
        cbar.set_label('Rhoa ohm.m')  # Set label for color bar
        # Save the current figure as a JPEG
        jpeg_filename = f"1D_Inversion_Nearest_to_references_{farmName}.jpeg"
        fig.savefig(jpeg_filename, dpi=300, bbox_inches='tight')  

        pdf_pages.savefig()
        pdf_pages.close()

   
#%% plot_3d_invResult
    
    
    def plot_3d_invResult(self, filepath, farm_name, elev=25, azim=250):
        zone = 33
        hemisphere = 'U'
    
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filepath, delimiter=';', skip_header=1)
    
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        Eutm, Nutm, zone_, letter_ = utm.from_latlon(y_values_Long, x_values_Lat, zone, hemisphere)
    
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
    
        # Prepare the Z values (depth levels)
        depth_levels = np.arange(0, 2.0, 0.1)  # 20 layers, each 10 cm deep
    
        # Create the 3D plot and save it to a PDF
        pdf_filename = f'3D_Inversion_Results_{farm_name}.pdf'
        with PdfPages(pdf_filename) as pdf:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
    
            # Collect all resistivity values for normalization
            all_rho_values = inv_res[:, 3:].flatten()
    
            # Plot the 3D scatter plot
            for layer_index, depth in enumerate(depth_levels):
                rho = inv_res[:, layer_index + 3]  # Assuming the rho values start from the fourth column
    
                # Offset the depth to create layers
                scatter = ax.scatter(
                    Eutm - x_offset, Nutm - y_offset, depth,
                    c=rho, cmap='Spectral_r', norm=LogNorm(vmin=10, vmax=1000), s=10, edgecolor='none'
                )
    
            ax.set_title(f'Inversion Results in 3D, {farm_name}')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            ax.set_zlabel('Depth (m)')
            ax.set_zlim(2, 0)  # Invert the z-axis to represent depth correctly
            ax.set_zticks(np.arange(0, 2.1, 0.1))  # From 0 to 2 meters with a step of 0.1 meters
    
            # Set equal scaling for Easting and Northing
            x_range = Eutm - x_offset
            y_range = Nutm - y_offset
            max_range = max(np.max(x_range) - np.min(x_range), np.max(y_range) - np.min(y_range))
    
            mid_x = (np.max(x_range) + np.min(x_range)) * 0.5
            mid_y = (np.max(y_range) + np.min(y_range)) * 0.5
    
            ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
            ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    
            # Set the view angle
            ax.view_init(elev, azim)  # Change elev and azim to desired values
    
            # Create color bar
            mappable = plt.cm.ScalarMappable(cmap='Spectral_r', norm=LogNorm(vmin=10, vmax=1000))
            mappable.set_array(all_rho_values)
            cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
            cbar.set_label('Resistivity (Ohm.m)')

            pdf.savefig(fig)
            plt.show()
            plt.close(fig)  # Close the figure after saving it to the PDF 
    
#%% plot_3d_invResult_section_maker
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.backends.backend_pdf import PdfPages
    import utm
    
    def plot_3d_invResult_section_maker(self, filepath, farm_name, elev=25, azim=250, easting_cutoff=None, northing_cutoff=None):
        zone = 33
        hemisphere = 'U'
    
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filepath, delimiter=';', skip_header=1)
    
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        Eutm, Nutm, zone_, letter_ = utm.from_latlon(y_values_Long, x_values_Lat, zone, hemisphere)
    
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
    
        # Apply the easting and northing cutoffs
        if easting_cutoff is not None:
            easting_mask = Eutm - x_offset >= easting_cutoff
        else:
            easting_mask = np.ones(Eutm.shape, dtype=bool)
            
        if northing_cutoff is not None:
            northing_mask = Nutm - y_offset >= northing_cutoff
        else:
            northing_mask = np.ones(Nutm.shape, dtype=bool)
    
        mask = easting_mask & northing_mask
        Eutm = Eutm[mask]
        Nutm = Nutm[mask]
        inv_res = inv_res[mask]
    
        # Prepare the Z values (depth levels)
        depth_levels = np.arange(0, 2.0, 0.1)  # 20 layers, each 10 cm deep
    
        # Create the 3D plot and save it to a PDF
        pdf_filename = f'3D_Inversion_Results_Cutoff, {farm_name}.pdf'
        with PdfPages(pdf_filename) as pdf:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
    
            # Collect all resistivity values for normalization
            all_rho_values = inv_res[:, 3:].flatten()
    
            # Plot the 3D scatter plot
            for layer_index, depth in enumerate(depth_levels):
                rho = inv_res[:, layer_index + 3]  # Assuming the rho values start from the fourth column
    
                # Offset the depth to create layers
                scatter = ax.scatter(
                    Eutm - x_offset, Nutm - y_offset, depth,
                    c=rho, cmap='Spectral_r', norm=LogNorm(vmin=10, vmax=1000), s=10, edgecolor='none'
                )
    
            ax.set_title(f'Inversion Results in 3D, Easting cut-off ={easting_cutoff}, Northing cut-off={northing_cutoff}, {farm_name}')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            ax.set_zlabel('Depth (m)')
            ax.set_zlim(2, 0)  # Invert the z-axis to represent depth correctly
            ax.set_zticks(np.arange(0, 2.1, 0.1))  # From 0 to 2 meters with a step of 0.1 meters
    
            # Set equal scaling for Easting and Northing
            x_range = Eutm - x_offset
            y_range = Nutm - y_offset
            max_range = max(np.max(x_range) - np.min(x_range), np.max(y_range) - np.min(y_range))
    
            mid_x = (np.max(x_range) + np.min(x_range)) * 0.5
            mid_y = (np.max(y_range) + np.min(y_range)) * 0.5
    
            ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
            ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    
            # Set the view angle
            ax.view_init(elev, azim)  # Change elev and azim to desired values
    
            # Create color bar
            mappable = plt.cm.ScalarMappable(cmap='Spectral_r', norm=LogNorm(vmin=10, vmax=1000))
            mappable.set_array(all_rho_values)
            cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
            cbar.set_label('Resistivity (Ohm.m)')
    
            pdf.savefig(fig)
            plt.show()
            plt.close(fig)  # Close the figure after saving it to the PDF
    
# %% compare ERi1 and rhoa1
    def subplot_compare_data_and_inversion_result(self, farmName, filepath_Original, filepath_Processed, inv_res_filepath, spacing_num=1):
        # Extract data for spacing 1
        data_processor = DataProcessing()  
        data_org = data_processor.read_data(filepath_Original)
        data_spacing1 = data_org[:, spacing_num + 2]  # Assuming data starts from the third column
        x_coords = data_org[:, 0]  # UTM Easting
        y_coords = data_org[:, 1]  # UTM Northing
        
        # Extract inversion result for the first layer
        inv_res = np.genfromtxt(inv_res_filepath, delimiter=';', skip_header=1)
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        zone = 33
        hemisphere = 'U'
        Eutm, Nutm, zone_, letter_ = utm.from_latlon(y_values_Long, x_values_Lat, zone, hemisphere)
        inv_res_first_layer = inv_res[:, 3]
        
        # Extract processed data
        data_proc = data_processor.read_data_lineIndexed(filepath_Processed)
        data_proc_spacing1 = data_proc[1:, spacing_num + 2]  # Assuming data starts from the third column
        x_coords_proc = data_proc[1:, 0]  # UTM Easting
        y_coords_proc = data_proc[1:, 1]  # UTM Northing
    
        # Calculate the dynamic figsize based on the range of Easting and Northing
        x_range = max(np.ptp(x_coords), np.ptp(Eutm), np.ptp(x_coords_proc))  # Peak-to-peak (max - min) range of Easting coordinates
        y_range = max(np.ptp(y_coords), np.ptp(Nutm), np.ptp(y_coords_proc))  # Peak-to-peak (max - min) range of Northing coordinates
    
        # Scaling factors to determine the figure size
        x_scale = 0.01  # Adjust this scaling factor to control width
        y_scale = 0.01  # Adjust this scaling factor to control height
    
        # Calculate the figure size dynamically
        fig_width = max(8, x_range * x_scale)  # Set a minimum width to avoid too small plots
        fig_height = max(6, y_range * y_scale)  # Set a minimum height to avoid too small plots
    
        # Create a subplot with three axes
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        fig.suptitle(f'Compare-ERi1-rhoa1 (Farm Name: {farmName})', fontsize=16)
    
        norm = colors.LogNorm(vmin=10, vmax=1000)
        # Plot data_org for spacing 1 on the left axis
        sc1 = ax1.scatter(x_coords, y_coords, c=data_spacing1, cmap='Spectral_r', s=0.7, norm=norm)
        ax1.set_title(f'Original data for Spacing {spacing_num}')
        ax1.set_xlabel('Easting (UTM)')
        ax1.set_ylabel('Northing (UTM)')
        ax1.set_aspect('equal')
    
        # Plot processed data for spacing 1 on the middle axis
        sc2 = ax2.scatter(x_coords_proc, y_coords_proc, c=data_proc_spacing1, cmap='Spectral_r', s=0.7, norm=norm)
        ax2.set_title(f'Processed data for Spacing {spacing_num}')
        ax2.set_xlabel('Easting (UTM)')
        ax2.set_ylabel('Northing (UTM)')
        ax2.set_aspect('equal')
    
        # Plot inversion result for the first layer on the right axis
        sc3 = ax3.scatter(Eutm, Nutm, c=inv_res_first_layer, cmap='Spectral_r', s=0.7, norm=norm)
        ax3.set_title('Inversion Result for First Layer')
        ax3.set_xlabel('Easting (UTM)')
        ax3.set_ylabel('Northing (UTM)')
        ax3.set_aspect('equal')
    
        # Add a color bar
        cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.05])  # Define position and size of colorbar
        cbar = plt.colorbar(sc3, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Resistivity (Ohm.m)')
    
        plt.tight_layout()
        plt.savefig(f'Compare-ERi1-rhoa1_{farmName}.jpg', dpi=300)  # Save the plot as a JPEG file with 300 dpi
        plt.show()
