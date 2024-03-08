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
    def __init__(self):
            pass
        
    #% 1D Inversion for all lines
    def inversion_1D_all_lines(self, filepath, amVec, farmName, Inv_File_name):
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
        # Get the current date in the format YY-MM-DD
        current_date = datetime.now().strftime('%y-%m-%d')
    
        spacing_labels = ['S1', 'S2', 'S3', 'S4', 'S5']
    
        data = self.read_data_lineIndexed(filepath)
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
            xy_coord_LatLong = []
    
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
                
            # convert the UTM coordinates back to latitude and longitude
            for coords in xy_coord_currentLine:
                Eutm, Nutm = coords
                latitude, longitude = to_latlon(Eutm, Nutm, 33, 'U')
                xy_coord_LatLong.append((latitude, longitude))
    
          
            # making an array for the inversion result for all data
            invResult_currentLine = np.hstack((np.array(xy_coord_LatLong), np.array(Stmodels), 
                                               np.reshape(chi2Vec_indiv, (-1, 1)), np.reshape(line_index, (-1, 1))))
            invResult_AllLines.append(invResult_currentLine)
            invResult_combined = np.concatenate(invResult_AllLines, axis=0)
            header = "X; Y;" + "; ".join([f"ERi{k + 1}" for k in range(len(modelInd))]) + "; Chi2;  line_index"
            
            # save the inv result with 2 different names
            result_file_1 = f'1Dinv_response_all_lines_{farmName}.csv'
            np.savetxt(result_file_1, invResult_combined, delimiter=';', fmt='%.7f', header=header)
            result_file_2_with_date = f'{Inv_File_name}_{current_date}.csv'
            np.savetxt(result_file_2_with_date, invResult_combined, delimiter=';', fmt='%.7f', header=header, comments='')
    
            # making an array for the forward response for all data    
            forward_resp_currentLine = np.hstack((np.array(xy_coord_LatLong), np.array(inv_response_all_soundings), 
                                               np.reshape(line_index, (-1, 1))))
            forward_resp_AllLines.append(forward_resp_currentLine)
            forward_resp_combined = np.concatenate(forward_resp_AllLines, axis=0)
            header_forward = "X; Y; " + "; ".join([f"ForwResp{k + 1}" for k in range(5)]) + ";  line_index"
            result_file_forward = f'1Dforward_response_all_lines_{farmName}.txt'
            np.savetxt(result_file_forward, forward_resp_combined, delimiter='\t', fmt='%.7f', header=header_forward)
            
            chi2_indiv_list.append(chi2Vec_indiv)
            Stmodels_list.append(Stmodels)
    
        return Stmodels_list, chi2_indiv_list
    
    
    def read_data_lineIndexed(self, filepath):
        EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma, line_index = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
        # EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, nan_data, Rho5, Gamma, BFI, lfd, Datum, Zeit = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
        data = np.column_stack((EdecDeg, NdecDeg, H, Rho1, Rho2, Rho3, Rho4, Rho5, Gamma, line_index))
        return data

    
    #% 1D inversion plotting for all lines 
    def plot_1d_inversion_results_allLines(self, filepath, amVec, farmName, data_type, thk):
        data = self.read_data_lineIndexed(filepath)
        data_no_nan = data[~np.isnan(data).any(axis=1)]
        data_filtered = data_no_nan[data_no_nan[:, -1] != 0] # remove individual points
        selected_lines = np.unique(data_filtered[:,-1])
        line_data = {}  # Dictionary to hold data for each line
    
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
        
                fig.suptitle(f'1D_Inversion (Farm Name: {farmName} , line_{int(line_num)})', fontsize=16)
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
    
    # Example usage:
    amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
    thk = np.ones(15) * 0.1
    
    farmName = 'Trebbin Wertheim_Processed'
    data_lineIndexed_file = f"data_lineIndexed_{farmName}.txt"
    
    
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
     
    # inv plot for original data
    data_type_org = 'Original'
    # plot_1d_inversion_results_allLines(data_lineIndexed_file, amVec, farmName_Original, data_type_org)
    
      
    #% inversion result subplot
    def subplot_inversion_results(self, filepath, farm_name):
        zone=33
        hemisphere='U'
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filepath, delimiter=';', skip_header=1)
        
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        Eutm, Nutm, _, _ = utm.from_latlon(x_values_Lat, y_values_Long, zone, hemisphere)
    
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
        
        pdf_filename = f'Inversion_Results_{farm_name}.pdf'
        with PdfPages(pdf_filename) as pdf:
            fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(11, 5))
            norm = LogNorm(10, 1000)
            for i, a in enumerate(ax.flat):
                rho = inv_res[:, i+2]  # Assuming the rho values start from the third column
                im = a.scatter(Eutm-x_offset, Nutm-y_offset, c=rho, s=0.1, norm=norm, cmap="Spectral_r", alpha=1)
                a.set_aspect(1.0)
                a.set_title("z={:.1f}-{:.1f}m".format(i*0.1, (i+1)*0.1))
        
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
    
    
    #% inversion result subplot chi2<20
    def subplot_inversion_limit_chi2(self, filepath, farm_name, Inv_File_name, chi2_limit=20, zone=33, hemisphere='U'):
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filepath, delimiter=';', skip_header=1)
        # Filter out rows where the Chi2 is greater than the limit
        inv_res_filtered = inv_res[inv_res[:, -2] <= chi2_limit]
    
        # Save the filtered inversion results to a text file
        header_inv = "X; Y;" + "; ".join([f"ERi{i}" for i in range(1, inv_res_filtered.shape[1]-3)]) + "; Chi2; line_index"
        current_date = datetime.now().strftime('%y-%m-%d')
        np.savetxt(f'{farm_name}_1Dinv_response_all_lines_combined_chi2_less_than_{chi2_limit}.txt', 
                   inv_res_filtered, delimiter=';', header=header_inv, fmt='%.7f', comments='')
        np.savetxt(f'{Inv_File_name}_{current_date}_chi2_lt_{chi2_limit}.csv', 
                   inv_res_filtered, delimiter=';', header=header_inv, fmt='%.7f', comments='')
    
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res_filtered[:, 0], inv_res_filtered[:, 1]
        Eutm, Nutm, _, _ = utm.from_latlon(x_values_Lat, y_values_Long, zone, hemisphere)
    
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
        pdf_filename = f'Inversion_Results_{farm_name}_chi2_lt_{chi2_limit}.pdf'
        with PdfPages(pdf_filename) as pdf:
    
            fig, ax = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(11, 5), 
                                   gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
        
            norm = LogNorm(10, 1000)
            for i, a in enumerate(ax.flat):
                rho = inv_res_filtered[:, i+2]  # Assuming the rho values start from the third column
                im = a.scatter(Eutm-x_offset, Nutm-y_offset, c=rho, s=0.1, norm=norm, cmap="Spectral_r", alpha=1)
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
        
            plt.suptitle(f'{farm_name} Inversion Results, chi2<{chi2_limit}', fontsize=16)
            pdf.savefig(fig)
    
            plt.show()

    
    #% inversion result multipage pdf
    def plot_inv_results_multipage_pdf(self, filename, farm_name, zone=33, hemisphere='U'):
        # Load data directly from the file using np.genfromtxt
        inv_res = np.genfromtxt(filename, delimiter=';', skip_header=1)
        # Latlong to UTM
        x_values_Lat, y_values_Long = inv_res[:, 0], inv_res[:, 1]
        Eutm, Nutm, _, _ = utm.from_latlon(x_values_Lat, y_values_Long, zone, hemisphere)
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
    
        # Create a multi-page PDF file
        pdf_filename = f'inversion_results_{farm_name}_MPpdf.pdf'
        with PdfPages(pdf_filename) as pdf:
            for i in range(15):  # Loop through all subplots
                fig, ax = plt.subplots(figsize=(8, 6))
                norm = LogNorm(10, 1000)
                rho = inv_res[:, i + 2]  # Assuming the rho values start from the third column
                im = ax.scatter(Eutm-x_offset, Nutm-y_offset, c=rho, s=0.8, norm=norm, cmap="Spectral_r", alpha=1)
                ax.set_title("z={:.1f}-{:.1f}m".format(i * 0.1, (i + 1) * 0.1))
                # Add color bar
                cbar = plt.colorbar(im, pad=0.02, shrink=0.8)
                cbar.set_label('Resistivity (Ohm.m)')
                # Set common labels
                ax.set_xlabel('Easting')
                ax.set_ylabel('Northing')
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
        Eutm, Nutm, _, _ = utm.from_latlon(x_values_Lat, y_values_Long, zone, hemisphere)
        x_offset = np.min(Eutm)
        y_offset = np.min(Nutm)
    
        # Create a multi-page PDF file
        pdf_filename = f'{farm_name}_Chi2_inversion_results.pdf'
        with PdfPages(pdf_filename) as pdf:
            fig, ax = plt.subplots(figsize=(8, 6))
            norm = LogNorm(1, 300)
            chi2 = inv_res[:, -2]  # Assuming the rho values start from the third column
            im = ax.scatter(Eutm-x_offset, Nutm-y_offset, c=chi2, s=0.2, norm=norm, cmap="Spectral_r", alpha=1)
            ax.set_aspect(1.0)
            ax.set_title(r"$\chi^2$")
            # Add color bar
            cbar = plt.colorbar(im, pad=0.02, shrink=0.5)
            cbar.set_label(r"$\chi^2$")
            # Set common labels
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            # Save the subplot into the PDF file
            plt.show()
    
            pdf.savefig(fig)
            plt.close(fig)



    def import_reference_points_from_kml(self, kml_file, filepath_original):
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
        refPoints = np.column_stack([ref_name, ref_eutm, ref_nutm])
    
        # Define the header
        header = ['Name', 'E', 'N']
    
        # Stack the header on top of the reference points
        ref_points_table = np.vstack((header, refPoints))
    
        # Save the reference points to a CSV file
        csv_filename = f'{filepath_original}_farm_reference_Points.csv'
        np.savetxt(csv_filename, ref_points_table, delimiter=';', fmt='%s')
        print(f'Reference points saved to {csv_filename}')
        return refPoints
