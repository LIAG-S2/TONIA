#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:56:43 2023

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
# %% import data

## Define file paths and farm names
# filepath = 'Geophilus_aufbereitet_2023-04-05_New.csv'
# farmName = 'aufbereitet'

# filepath = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName = 'Trebbin Wertheim'

# filepath = 'Trebbin Aussenschlag.csv'
# farmName = 'Trebbin Aussenschlag'

# filepath = 'BFD_Banane_Geophilus_221118_roh.csv'
# farmName = 'Beerfelde Banane'

# filepath = 'BFD_Fettke_Geophilus_roh.csv'
farmName = 'Beerfelde Fettke'
filepath_Processed = 'BFD_Fettke_Geophilus_roh.csv'
farmName_Processed = 'Beerfelde_Fettke_Processed data'
filepath_original = 'Beerfelde 2022_10_21-original.csv'
farmName_original = 'Beerfelde_Fettke_Original data'

# filepath = 'BDW_Hoben_Geophilus_221020_roh.csv'
# farmName = 'Brodowin Hoben'

# filepath = 'BDW_Seeschlag_Geophilus_roh_221020 - Copy.csv'
# farmName = 'Brodowin Seeschlag'

# filepath = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName = 'Boossen_1211'

# filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName = 'Boossen_1601'
#%% Functions
filepath = filepath_original
# Function to read the processed data
def read_processed_data(filepath):
    EdecDeg_P, NdecDeg_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Rho5_P, Gamma_P, BFI, lfd, Datum, Zeit = np.genfromtxt(
        filepath, skip_header=1, delimiter=',', unpack=True)
    Eutm_P, Nutm_P, zone, letter = utm.from_latlon(NdecDeg_P, EdecDeg_P)
    farmName_Processed = 'Beerfelde Fettke Processed data'
    np.savetxt(f'{farmName_Processed} farm_data.csv', np.column_stack((Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P)), delimiter=';', fmt='%s')
    data_processed = np.column_stack((Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P))
    return data_processed

# Function to read the original data
def read_original_data(filepath):
    EdecDeg_O, NdecDeg_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, nan_data, Rho5_O, Gamma_O = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
    Eutm_O, Nutm_O, zone, letter = utm.from_latlon(NdecDeg_O, EdecDeg_O)
    farmName_original = 'Beerfelde_Fettke_Original data'
    np.savetxt(f'{farmName_original}.txt', np.column_stack((Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O)), delimiter=';', fmt='%s')
    data_original = np.column_stack((Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O))
    num_spacing = data_original.shape[1]  # 5 spacing in Geophilus system

    return data_original


# Function to plot original data's areal plot
def plot_original_data(data_original, spacing_labels, farmName, refPoints_df, cmap='jet', vmin=10, vmax=1000):
    # Create a multi-page PDF file to save the plots
    pdf_filename = f"Original_Data_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    for i, spacing in enumerate(spacing_labels):
        # Create a new figure for each spacing plot
        plt.figure(figsize=(8, 6))
        
        # Extract UTM coordinates (Easting and Northing) for the current spacing
        x_coords = data_original[:, 0]  # UTM Easting
        y_coords = data_original[:, 1]  # UTM Northing
        
        # Draw the areal plot for the original data points
        sc = plt.scatter(x_coords, y_coords, s=0.7, c=data_original[:, i + 3], cmap='jet', norm=LogNorm(vmin=vmin, vmax=vmax))
        
        # Plot the reference points on the subplot
        plt.scatter(refPoints_df['E'], refPoints_df['N'], c='red', marker='x', s=50, label='Reference Points')
        
        # Annotate each reference point with its corresponding index (number)
        for j, txt in enumerate(refPoints_df['Name']):
            plt.annotate(txt, (refPoints_df['E'][j], refPoints_df['N'][j]), fontsize=13, color='black')

        # colorbar
        plt.colorbar(sc)

        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Original Data - {spacing} - {farmName}', y=1.03)

        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()

        # Close the current figure
        plt.close()

    # Close the PDF file
    pdf_pages.close()


# Class for line detection
class LineDetector:
    def __init__(self, minDist=200., maxDeviation=5.):
        self.minDist = minDist
        self.maxDeviation = maxDeviation

    def detect_lines(self, rx, ry):
        """Split data into lines based on distance and angle deviation."""
        dummy = np.zeros_like(rx, dtype=int)
        line = np.zeros_like(rx, dtype=int)
        li = 0
        for ri in range(1, len(rx)):
            dummy[ri-1] = li
            dist = np.sqrt((rx[ri]-rx[ri-1])**2 + (ry[ri]-ry[ri-1])**2)
            angle_deviation = np.arctan2(ry[ri] - ry[ri-1], rx[ri] - rx[ri-1]) * 180 / np.pi
            if dist > self.minDist or angle_deviation > self.maxDeviation:
                li += 1
            else:
                dummy[ri] = li
        dummy[-1] = li

        return self._sort_lines(rx, ry, line, dummy)

    def _sort_lines(self, rx, ry, line, dummy):
        """
        Sort line elements by Rx or Ry coordinates.Beerfelde
        """
        means = []
        for li in np.unique(dummy):
            means.append(np.mean(ry[dummy==li], axis=0))

        lsorted = np.argsort(means)
        for li, lold in enumerate(lsorted):
            line[dummy == lold] = li + 1

        return line

# Function to perform Isolation Forest on original data
def isolation_forest_outlier_detection(data_original, num_spacing, refPoints_df):
    # Set a fixed random seed
    np.random.seed(42)
    # Create an array to store the Isolation Forest results
    resistivity_data = data_original[:,3:8]
    num_spacing = resistivity_data.shape[1]  # 5 spacing in Geophilus system# should be outside of the function

    Isolation_Forest = np.empty((len(resistivity_data), num_spacing + 4))
    Isolation_Forest[:] = np.nan

    # Store the (X,Y) UTM coordinates, H_original, and Gamma_original
    Isolation_Forest[:, 0] = data_original[:,0]
    Isolation_Forest[:, 1] = data_original[:,1]
    Isolation_Forest[:, 2] = data_original[:,2]
    Isolation_Forest[:, -1] = data_original[:,8]

    num_outliers = np.zeros(num_spacing)  # Array to store the number of outliers per spacing

    # Create a multi-page PDF file to save the plots
    pdf_filename = f"Isolation_Forest_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)
    
    
    for spacing in range(num_spacing):
        # Original data
        resistivity_values_original = np.log(resistivity_data[:, spacing])

        # Outlier detection using Isolation Forest on original data
        isolation_forest = IsolationForest()
        outliers_iforest = isolation_forest.fit_predict(resistivity_values_original.reshape(-1, 1))

        # Count the number of outliers
        num_outliers[spacing] = np.sum(outliers_iforest == -1)

        # Store the Isolation Forest results for each spacing
        selected_values = np.exp(resistivity_values_original[outliers_iforest == 1])
        Isolation_Forest[outliers_iforest == 1, spacing + 3] = selected_values
        
            # Draw the areal plot for the Isolation Forest data points
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(Isolation_Forest[:, 0], Isolation_Forest[:, 1],
                         s=0.7, c=Isolation_Forest[:, spacing + 3], cmap='jet', norm=LogNorm(vmin=10, vmax=1000))
        
        # Plot the reference points on the subplot
        plt.scatter(refPoints_df['E'], refPoints_df['N'], c='red', marker='x', s=50, label='Reference Points')
    
        # Annotate each reference point with its corresponding index (number)
        for j, txt in enumerate(refPoints_df['Name']):
            plt.annotate(txt, (refPoints_df['E'][j], refPoints_df['N'][j]), fontsize=13, color='black')
    
        # colorbar
        plt.colorbar(sc)
    
        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Isolation Forest - Spacing {spacing + 1} - {farmName}', y=1.03)
    
        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()
    
        # Close the current figure
        plt.close()
    
    # Close the PDF file
    pdf_pages.close()
        
    # Save the Isolation Forest data to a single text file
    output_filename = f"Isolation_{farmName.replace(' ', '')}.txt"
    header = 'x\ty\th\t' + '\t'.join([f"Rho Spacing {i+1}" for i in range(num_spacing)]) + '\tGamma'
    np.savetxt(output_filename, Isolation_Forest, delimiter='\t', header=header)

    # Print the number of outliers and non-outliers per spacing
    for spacing in range(num_spacing):
        print(f"Spacing {spacing + 1}: Outliers = {int(num_outliers[spacing])}, Non-Outliers = {int(len(resistivity_data) - num_outliers[spacing])}")

    return Isolation_Forest

# Function to perform harmfit on Isolation Forest result
def harmfit_without_nan(data, nc=30, error=0.01):
    # Filter out invalid data (nan values)
    valid_indices = np.isfinite(data[:,2])
    valid_data = data[:,2][valid_indices]

    xline = np.sqrt((data[:,0]-data[:,0][0])**2+(data[:,1]-data[:,1][0])**2)
    # Perform harmfit on the valid data
    harmfit_result, _ = harmfit(np.log(valid_data), x=xline[valid_indices], nc=nc, error=error)

    # Create a new array to store the harmfit result with nan values restored
    harmfit_result_with_nan = np.empty(len(data))
    harmfit_result_with_nan[:] = np.nan
    harmfit_result_with_nan[valid_indices] = np.exp(harmfit_result)

    x_harmfitted = data[:,0].copy()
    y_harmfitted = data[:,1].copy()

    return x_harmfitted[valid_indices], y_harmfitted[valid_indices], np.exp(harmfit_result)


# Function to harmfit on Isolation Forest
def harmfit_on_IF(Isolation_Forest, spacing_labels, selected_lines, line_indices, farmName, refPoints_df, cmap='jet', vmin=10, vmax=1000):
    # Create a multi-page PDF file to save the plots
    pdf_filename = f"Harmfit_Isolation_forest_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)
    nc=30
    # Iterate over each spacing
    for i, spacing in enumerate(spacing_labels):
        # Create a new figure for each spacing plot
        plt.figure(figsize=(8, 6))
        x_all = []
        y_all = []
        harmfitted_spacing_data_list_all = []  # List to store harmfitted data for all lines and the current spacing

        # Iterate over each selected line
        for line_num in selected_lines:
            point_indices_of_interest = np.where(line_indices == line_num)[0]
            
            # Convert point_indices_of_interest to integers
            point_indices_of_interest = point_indices_of_interest.astype(int)
            
            # Get the UTM coordinates (Easting and Northing) for the current line
            x_coords = Isolation_Forest[point_indices_of_interest, 0]  # UTM Easting
            y_coords = Isolation_Forest[point_indices_of_interest, 1]  # UTM Northing
            h = Isolation_Forest[point_indices_of_interest, 2]  #
            
            # Get the resistivity values for the current spacing and line
            resistivity_spacing_line = Isolation_Forest[point_indices_of_interest, i + 3]  # should have x and y
            
            # Perform harmfit on the Isolation Forest resistivity data for the current line and spacing
            x_harmfitted, y_harmfitted, harmfitted_IF_data = harmfit_without_nan(
                np.column_stack((x_coords, y_coords, resistivity_spacing_line)),
                nc=nc,
                error=0.01)
            
            x_all.extend(x_harmfitted)
            y_all.extend(y_harmfitted)
            harmfitted_spacing_data_list_all.extend(harmfitted_IF_data)
            
        # Draw the areal plot for all lines together for the current spacing
        sc = plt.scatter(x_all, y_all,
                         s=0.7, c=harmfitted_spacing_data_list_all, cmap='jet', norm=LogNorm(vmin=10, vmax=1000))
        
        # Plot the reference points on the subplot
        plt.scatter(refPoints_df['E'], refPoints_df['N'], c='red', marker='x', s=50, label='Reference Points')
        
        # Annotate each reference point with its corresponding index (number)
        for j, txt in enumerate(refPoints_df['Name']):
            plt.annotate(txt, (refPoints_df['E'][j], refPoints_df['N'][j]), fontsize=13, color='black')
        
        # colorbar
        plt.colorbar(sc)
        
        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Harmfit on Isolation forest - {spacing} - {farmName}' , y=1.03)
        
        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()
        
        # Close the current figure
        plt.close()

        # Save the harmfitted data for the current spacing in a text file
        harmfitted_spacing_file = f"harmfit_spacing_{i+1} - {farmName}.txt"
        np.savetxt(harmfitted_spacing_file, np.column_stack((x_all, y_all, harmfitted_spacing_data_list_all)), delimiter='\t')

    # Close the PDF file
    pdf_pages.close()

    return harmfit_on_IF


# Main function
def main():
    data_file = "Beerfelde 2022_10_21-original.csv"
    farmName = 'Beerfelde Fettke'
    spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']
    # Read original data
    data_original = read_original_data(data_file)
    # Create LineDetector instance with desired parameters
    line_detector = LineDetector(minDist=200, maxDeviation=155)  
    # Call the detect_lines method on the instance
    line_indices = line_detector.detect_lines(data_original[:,0], data_original[:,1])
    # Detect lines
    line_indices = line_detector.detect_lines(data_original[:,0], data_original[:,1])
    # Get unique line indices and their counts
    unique_lines, counts = np.unique(line_indices, return_counts=True)
    # Filter lines with more than 10 points
    selected_lines =  unique_lines[counts >= 10]
    
    # Print detected lines with more than 10 points
    num_lines = len(selected_lines)
    num_unvalid_lines = len(unique_lines) - len(selected_lines)
    print("Number of valid lines detected: ", num_lines)
    print("Number of unvalid lines (containing less than 10 points): ", num_unvalid_lines) 
    
    # import reference points from kml file
    kmlFile = "BFD_Fettke_Referenzpunkte_gemessen.kml"
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    gdf = gpd.read_file(kmlFile) #GeoDataFrame object (ref names & geometry,in decimaldegree)
    refLat = np.array(gdf['geometry'].y)
    refLon = np.array(gdf['geometry'].x)
    refEutm, refNutm, refzone, refletter  = utm.from_latlon(refLat, refLon) # convert E&N from DecimalDegree to UTM
    refName = np.array(gdf['Name'])
    refPoints = np.column_stack([refName, refEutm, refNutm])# RefPointNames+geometry,in lat/long
    header = ['Name', 'E', 'N']
    refPointsTable = np.vstack((header, refPoints))
    np.savetxt(f'{farmName} farm_reference_Points.csv', refPointsTable, delimiter=';', fmt='%s')
    refPoints_df = pd.DataFrame(refPoints, columns=['Name', 'E', 'N'])
    
    # Plot original data's areal plot
    plot_original_data(data_original, spacing_labels, farmName, refPoints_df)

    # Perform Isolation Forest
    num_spacing = len(spacing_labels)
    Isolation_Forest = isolation_forest_outlier_detection(data_original, num_spacing, refPoints_df)  # Store the Isolation Forest data
    
    # Perform harmfit on Isolation Forest result and draw areal plots
    harmfit_on_IF(Isolation_Forest, spacing_labels, selected_lines, line_indices, farmName, refPoints_df)


if __name__ == "__main__":
    main()

#%% combine harmfit data with nan values and make an array 

import numpy as np

def combined_harmfit_data(farmName):
    # Read the Isolation data
    isolation_data = np.loadtxt(f"Isolation_{farmName.replace(' ', '')}.txt", delimiter='\t', skiprows=1)
    isolation_data = isolation_data[1:, :]  # Remove the first row
    x_isolation = isolation_data[:, 0]
    y_isolation = isolation_data[:, 1]
    h_isolation = isolation_data[:, 2]
    isolation_values = isolation_data[:, 3:-1]  # Exclude the first three columns and the last column (Gamma)

    # Read the original data
    original_data = np.loadtxt(f'{farmName_original}.txt', delimiter=';', skiprows=1)
    x_original = original_data[:, 0]
    y_original = original_data[:, 1]
    h_original = original_data[:, 2]
    original_values = original_data[:, 3:-1]  # Exclude the first three columns and the last column (Gamma)
        
    # Create a dictionary to store harmfit data
    harmfit_data = {}
    
    # Read and combine harmfit data from all harmfit files
    num_spacing = 5  # Change this to the actual number of spacing
    
    # Create the combined array with NaN values
    combined_harmfit_array = np.empty((len(x_original), 3 + num_spacing))
    combined_harmfit_array[:, :] = np.nan
    combined_harmfit_array[:, 0] = x_original
    combined_harmfit_array[:, 1] = y_original
    combined_harmfit_array[:, 2] = h_original

    nan_counts = np.zeros(3 + num_spacing)
    non_nan_counts = np.zeros(3 + num_spacing)
    
    for i in range(num_spacing):
        harmfit_file = f"harmfit_spacing_{i+1} - {farmName}.txt"
        harmfit_values = np.loadtxt(harmfit_file, delimiter='\t', skiprows=1)
        #print(harmfit_values.shape)

        # Extract x and y values from harmfit data
        x_harmfit = harmfit_values[:, 0]
        y_harmfit = harmfit_values[:, 1]

        # Find the common x and y values between original data and harmfit data
        common_indices_original = np.where(np.logical_and(
            np.isin(x_original, x_harmfit),
            np.isin(y_original, y_harmfit)))[0]

        # Fill in harmfit values for common x and y
        for idx in common_indices_original:
            x_idx = np.where(x_harmfit == x_original[idx])[0]
            y_idx = np.where(y_harmfit == y_original[idx])[0]
            common_idx = np.intersect1d(x_idx, y_idx)
            if common_idx.size > 0:
                combined_harmfit_array[idx, i + 3] = harmfit_values[common_idx[0], 2]

        harmfit_data[f'spacing_{i+1}'] = combined_harmfit_array[:, i + 3]

        # Count NaN and non-NaN values in the column
        nan_counts[i + 3] = np.sum(np.isnan(combined_harmfit_array[:, i + 3]))
        non_nan_counts[i + 3] = np.sum(~np.isnan(combined_harmfit_array[:, i + 3]))


    # Save the combined data to a new text file
    combined_filename = f"Combined_harmfit_array_{farmName}.txt"
    header = 'x\ty\th\t' + '\t'.join([f"Harmfit Spacing {i+1}" for i in range(num_spacing)])
    np.savetxt(combined_filename, combined_harmfit_array, delimiter='\t', header=header)

    return combined_harmfit_array

# call combine_and_save_data
combined_harmfit_data(farmName)

#%% compare the harmfitted Isolation Forest (HFIF) data with the original data for each line and spacing
def plot_harmfit_vs_original(data_file, farmName):
    spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']
    nc=30 #to be removed later
    # Read original data
    data_original = read_original_data(data_file)
    Eutm_O, Rho_original = data_original[:, 0], data_original[:, 3:8]

    # Define the color map for spacing plots
    colors_ = ['b', 'g', 'r', 'c', 'm']

    combined_Harmfit = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
    x_combined_Harmfit = combined_Harmfit[:, 0]
    y_combined_Harmfit = combined_Harmfit[:, 1]

    # Create LineDetector instance with desired parameters
    line_detector = LineDetector(minDist=200, maxDeviation=155)

    # Detect lines
    line_indices = line_detector.detect_lines(x_combined_Harmfit, y_combined_Harmfit)

    # Get unique line indices and their counts
    unique_lines, counts = np.unique(line_indices, return_counts=True)

    # Filter lines with more than 10 points
    selected_lines = unique_lines[counts >= 10]

    pdf_filename = f"Harmfit_vs_Original-{farmName}, nc={nc}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    for line_idx, line_num in enumerate(selected_lines, start=1):
        point_indices_of_interest = np.where(line_indices == line_num)[0]

        # plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        for j, spacing in enumerate(spacing_labels):
            combined_Harmfit = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
            line_indices_HFIF = line_detector.detect_lines(combined_Harmfit[:, 0], combined_Harmfit[:, 1])

            # Get unique line indices and their counts
            unique_lines_HFIF, counts_HFIF = np.unique(line_indices_HFIF, return_counts=True)

            ax.plot(
                combined_Harmfit[:, 0][point_indices_of_interest],
                combined_Harmfit[:, j + 3][point_indices_of_interest],
                label=f'harmfit IF data - {spacing}',
                color=colors_[j]
            )

        for j, rho in enumerate(Rho_original.T):
            ax.scatter(
                Eutm_O[point_indices_of_interest],
                rho[point_indices_of_interest],
                label=f'Original Data - {spacing_labels[j]}',
                color=colors_[j],
                marker='.',
                s=10
            )

        ax.set_xlabel('Easting (UTM)')
        ax.set_ylabel('Resistivity')
        ax.set_title(f'Line {line_idx} - Harmfit vs Original data {farmName}, nc={nc}')
        ax.set_yscale('log')
        ax.legend()

        plt.tight_layout()
        plt.show()
        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()

# Call the function with appropriate arguments
data_file = "Beerfelde 2022_10_21-original.csv"
farmName = 'Beerfelde Fettke'
plot_harmfit_vs_original(data_file, farmName)

#%% plot original, IF and Harmfit on IF data together
def plot_combined_subplots(data_file, farmName):
    spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

    # Read original data
    data_original = read_original_data(data_file)
    Eutm_O, Nutm_O, Rho_original = data_original[:, 0], data_original[:, 1], data_original[:, 3:8]

    # Define the color map for spacing plots
    colors_ = ['b', 'g', 'r', 'c', 'm']

    combined_Harmfit = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
    x_combined_Harmfit = combined_Harmfit[:, 0]
    y_combined_Harmfit = combined_Harmfit[:, 1]

    # Load Isolation Forest data
    isolation_data = np.genfromtxt(f"Isolation_{farmName.replace(' ', '')}.txt", delimiter='\t', skip_header=1)
    x_isolation = isolation_data[:, 0]
    y_isolation = isolation_data[:, 1]
    rho_isolation = isolation_data[:, 3:8]

    # Create LineDetector instance with desired parameters
    line_detector = LineDetector(minDist=200, maxDeviation=155)

    # Detect lines
    line_indices = line_detector.detect_lines(x_combined_Harmfit, y_combined_Harmfit)

    # Get unique line indices and their counts
    unique_lines, counts = np.unique(line_indices, return_counts=True)

    # Filter lines with more than 10 points
    selected_lines = unique_lines[counts >= 10]

    pdf_filename = f"Combined_Subplots-{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    for j, spacing in enumerate(spacing_labels):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        for line_idx, line_num in enumerate(selected_lines, start=1):
            point_indices_of_interest = np.where(line_indices == line_num)[0]

            combined_Harmfit = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
            line_indices_HFIF = line_detector.detect_lines(combined_Harmfit[:, 0], combined_Harmfit[:, 1])

            # Get unique line indices and their counts
            unique_lines_HFIF, counts_HFIF = np.unique(line_indices_HFIF, return_counts=True)

            # Plot original data
            sc1 = axs[0].scatter(
                Eutm_O[point_indices_of_interest],
                Nutm_O[point_indices_of_interest],
                c=Rho_original[point_indices_of_interest, j],
                cmap='jet',
                norm=LogNorm(vmin=10, vmax=1000),
                marker='.',
                s=10
            )

            # Plot Isolation Forest data
            sc2 = axs[1].scatter(
                x_isolation[point_indices_of_interest],
                y_isolation[point_indices_of_interest],
                c=rho_isolation[point_indices_of_interest, j],
                cmap='jet',
                norm=LogNorm(vmin=10, vmax=1000),
                marker='.',
                s=10
            )

            # Plot Harmfitted on Isolation Forest data
            harmfit_data = combined_Harmfit[:, j + 3][point_indices_of_interest]
            valid_indices = np.isfinite(harmfit_data)
            sc3 = axs[2].scatter(
                combined_Harmfit[:, 0][point_indices_of_interest][valid_indices],
                combined_Harmfit[:, 1][point_indices_of_interest][valid_indices],
                c=harmfit_data[valid_indices],
                cmap='jet',
                norm=LogNorm(vmin=10, vmax=1000),
                marker='.',
                s=10
            )

        for ax, title in zip(axs, ['Original Data', 'Isolation Forest Data', 'Harmfitted on Isolation Forest Data']):
            ax.set_xlabel('Easting (UTM)')
            ax.set_ylabel('Northing (UTM)')
            ax.set_yscale('log')
            ax.set_title(f'{title} - Spacing {spacing}')
            plt.colorbar(sc1, ax=ax)

        plt.tight_layout()
        plt.show()
        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()

# Call the function with appropriate arguments
data_file = "Beerfelde 2022_10_21-original.csv"
farmName = 'Beerfelde Fettke'
plot_combined_subplots(data_file, farmName)
#%% plot 1-original data, 2-processed data, 3-IF and 4-Harmfit on IF data together
def plot_combined_subplots(data_file, filepath_Processed, farmName):
    spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

    # Read original data
    data_original = read_original_data(data_file)
    Eutm_O, Nutm_O, Rho_original = data_original[:, 0], data_original[:, 1], data_original[:, 3:8]

    # Define the color map for spacing plots
    colors_ = ['b', 'g', 'r', 'c', 'm']

    combined_Harmfit = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
    x_combined_Harmfit = combined_Harmfit[:, 0]
    y_combined_Harmfit = combined_Harmfit[:, 1]

    # Load Isolation Forest data
    isolation_data = np.genfromtxt(f"Isolation_{farmName.replace(' ', '')}.txt", delimiter='\t', skip_header=1)
    x_isolation = isolation_data[:, 0]
    y_isolation = isolation_data[:, 1]
    rho_isolation = isolation_data[:, 3:8]

    # Load processed data
    processed_data = read_processed_data(filepath_Processed)
    Eutm_P, Nutm_P, Rho_processed = processed_data[:, 0], processed_data[:, 1], processed_data[:, 3:8]

    # Create LineDetector instance with desired parameters
    line_detector = LineDetector(minDist=200, maxDeviation=155)

    # Detect lines
    line_indices = line_detector.detect_lines(x_combined_Harmfit, y_combined_Harmfit)

    # Get unique line indices and their counts
    unique_lines, counts = np.unique(line_indices, return_counts=True)

    # Filter lines with more than 10 points
    selected_lines = unique_lines[counts >= 10]

    pdf_filename = f"Combined_Subplots-{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    for j, spacing in enumerate(spacing_labels):
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))  # Adjust the figsize as needed

        for line_idx, line_num in enumerate(selected_lines, start=1):
            point_indices_of_interest = np.where(line_indices == line_num)[0]

            combined_Harmfit = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
            line_indices_HFIF = line_detector.detect_lines(combined_Harmfit[:, 0], combined_Harmfit[:, 1])

            # Get unique line indices and their counts
            unique_lines_HFIF, counts_HFIF = np.unique(line_indices_HFIF, return_counts=True)

            # Plot original data
            sc1 = axs[0].scatter(
                Eutm_O[point_indices_of_interest],
                Nutm_O[point_indices_of_interest],
                c=Rho_original[point_indices_of_interest, j],
                cmap='jet',
                norm=LogNorm(vmin=10, vmax=1000),
                marker='.',
                s=10
            )

            # Plot Processed data
            if len(point_indices_of_interest) > 0:
                processed_indices = point_indices_of_interest % len(Eutm_P)  # Adjust indexing loop
                sc2 = axs[1].scatter(
                    Eutm_P[processed_indices],
                    Nutm_P[processed_indices],
                    c=Rho_processed[processed_indices, j],
                    cmap='jet',
                    norm=LogNorm(vmin=10, vmax=1000),
                    marker='.',
                    s=10
                )


            # Plot Isolation Forest data
            sc3 = axs[2].scatter(
                x_isolation[point_indices_of_interest],
                y_isolation[point_indices_of_interest],
                c=rho_isolation[point_indices_of_interest, j],
                cmap='jet',
                norm=LogNorm(vmin=10, vmax=1000),
                marker='.',
                s=10
            )

            
            # Plot Harmfitted on Isolation Forest data
            harmfit_data = combined_Harmfit[:, j + 3][point_indices_of_interest]
            valid_indices = np.isfinite(harmfit_data)
            sc4 = axs[3].scatter(
                combined_Harmfit[:, 0][point_indices_of_interest][valid_indices],
                combined_Harmfit[:, 1][point_indices_of_interest][valid_indices],
                c=harmfit_data[valid_indices],
                cmap='jet',
                norm=LogNorm(vmin=10, vmax=1000),
                marker='.',
                s=10
            )


            
        for ax, title in zip(axs, ['Original Data', 'Processed Data', 'Isolation Forest Data', 'Harmfitted on Isolation Forest Data']):
            ax.set_xlabel('Easting (UTM)')
            ax.set_ylabel('Northing (UTM)')
            ax.set_title(f'{title}')
            fig.colorbar(ax.get_children()[0], ax=ax, label=r'$\rho$')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect values as needed
        fig.suptitle(f"Areal Plots {spacing} - {farmName} farm", fontsize=16, y=0.98)  # Adjust fontsize and y position as needed

        plt.show()
        pdf_pages.savefig(fig)
        plt.close(fig)

    pdf_pages.close()




# Call the function with appropriate arguments

filepath_Processed = 'BFD_Fettke_Geophilus_roh.csv'

data_file = "Beerfelde 2022_10_21-original.csv"
farmName = 'Beerfelde Fettke'
plot_combined_subplots(data_file, filepath_Processed, farmName)

# %%






# %% Find nearest dataPoints to refPoints
f'{farmName} farm_reference_Points.csv'
import numpy as np
import pandas as pd

def find_nearest_points(farmName, data):
    refpoints = f'{farmName} farm_reference_Points.csv'
    RefPoints = pd.read_csv(refpoints, delimiter=";")

    nearest_points_list = []

    for point in RefPoints.itertuples():
        array = data
        dist = np.sqrt((array[:, 0] - point[2]) ** 2 + (array[:, 1] - point[3]) ** 2)
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index - 5, dist_index + 5)

        data_nearest = array[nearestArray]   # data of individual closest points
        nearest_points_list.append(data_nearest)

    return nearest_points_list, RefPoints

def mean_nearest_points(nearest_points_list):
    meanDataVal = []
    nan_reference_points = []  # List to store indices of reference points with NaN mean values

    for i, data_nearest in enumerate(nearest_points_list):
        # Process data
        Eutm, Nutm, Rho1, Rho2, Rho3, Rho4, Rho5 = data_nearest[:, [0, 1, 3, 4, 5, 6, 7]].T

        # Check if any of the mean values is NaN
        if np.any(np.isnan([np.nanmean(Eutm), np.nanmean(Nutm), np.nanmean(Rho1), 
                            np.nanmean(Rho2), np.nanmean(Rho3), np.nanmean(Rho4), 
                            np.nanmean(Rho5)])):
            nan_reference_points.append(i)  # Store the index of the reference point with NaN mean values

        MeanOfNearestPoints = np.column_stack((np.nanmean(Eutm), np.nanmean(Nutm), \
                                               np.nanmean(Rho1), np.nanmean(Rho2), \
                                               np.nanmean(Rho3), np.nanmean(Rho4), \
                                               np.nanmean(Rho5)
                                              ))        
        meanDataVal.append(MeanOfNearestPoints)

    return np.array(meanDataVal), nan_reference_points


#  Find ref points and their mean values
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']
harmfit_file = np.genfromtxt( f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
nearestPoints, refPoints = find_nearest_points(farmName, harmfit_file)
mean_nearest_p, nan_reference_points = mean_nearest_points(nearestPoints)
mean_nearest_p_array = mean_nearest_p.reshape(-1, mean_nearest_p.shape[-1])

if len(nan_reference_points) == len(refPoints):
    # If all reference points have NaN mean values
    print(f"For all reference points, there are no accepted data nearby for inversion.")
else:
    # Store the reference points with NaN mean values to a CSV file
    nan_reference_points_df = refPoints.iloc[nan_reference_points]
    nan_reference_points_df.to_csv(f'{farmName}-nanReferencePoints.csv', index=False, sep=';')

    # Print the reference points with NaN mean values
    print(f"For these reference points, there are no accepted data nearby for inversion:")
    print(nan_reference_points_df)
    
# Remove reference points with NaN mean values from refPoints DataFrame
refPoints_cleaned = refPoints.drop(nan_reference_points, axis=0)

# Save the mean data of nearest points to a CSV file without NaN rows
header = ['Eutm', 'Nutm', 'Rho1', 'Rho2', 'Rho3', 'Rho4', 'Rho6']
np.savetxt(f'meanNearestPoints-noNaN - {farmName}.txt', mean_nearest_p_array[~np.any(np.isnan(mean_nearest_p_array), axis=1)], delimiter=';', header=';'.join(header), fmt='%.6f')

# Save the original mean data of nearest points to another CSV file with NaN rows
np.savetxt(f'meanNearestPoints-withNaN- {farmName}.txt', mean_nearest_p_array, delimiter=';', header=';'.join(header), fmt='%.6f')









# %% import data for inversion
import pandas as pd

refPoints = pd.read_csv(f'{farmName} farm_reference_Points.csv', delimiter=';')
# refName = refPoints_cleaned['Name'].astype(int) # refPoint names
refName = refPoints['Name'].astype(int) # refPoint names
#meanNearestPoints_harmfit_on_IF = np.loadtxt(f'meanNearestPoints-noNaN - {farmName}.txt', delimiter=';') 
meanNearestPoints_harmfit_on_IF = np.loadtxt(f'meanNearestPoints-withNaN- {farmName}.txt', delimiter=';') 
#meanNearestPoints_processed = np.loadtxt(f'{farmName}-meanNearestPoints_processed.csv', delimiter=';')
#data_O = np.loadtxt(f'{farmName_original} farm_data.csv', delimiter=';')
#harmfit_data = np.loadtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skiprows=1)
harmfit_data = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)

#nearestArray_harmfit = pd.read_csv(f'{farmName}-nearestArray_harmfit.csv', delimiter=',') # data of individual closest points
#nearestArray_processed = pd.read_csv(f'{farmName}-nearestArray_processed.csv', delimiter=',') # data of individual closest points
# %% filter_acceptable_refpoints (Unacceptables have only NAN values in their vicinit)
import pandas as pd

def filter_acceptable_refpoints(refpoints_file, unacceptable_refpoints_file):
    # Read the reference points and unacceptable reference points CSV files
    RefPoints = pd.read_csv(refpoints_file, delimiter=";")
    unacceptable_RefPoints = pd.read_csv(unacceptable_refpoints_file, delimiter=";")

    # Extract the column name of the reference points from the DataFrame
    refpoint_column_name = RefPoints.columns[0]

    # Extract the column name of the unacceptable reference points from the DataFrame
    unacceptable_refpoint_column_name = unacceptable_RefPoints.columns[0]

    # Create a list of unacceptable reference points
    unacceptable_points_list = unacceptable_RefPoints[unacceptable_refpoint_column_name].tolist()

    # Filter out the acceptable reference points by removing the unacceptable ones
    acceptable_refpoints = RefPoints[~RefPoints[refpoint_column_name].isin(unacceptable_points_list)]

    # Get the E and N coordinates of unacceptable reference points
    unacceptable_refpoints_coords = RefPoints[RefPoints[refpoint_column_name].isin(unacceptable_points_list)]
    
    # Concatenate acceptable and unacceptable reference points
    all_refpoints = RefPoints


    return acceptable_refpoints, unacceptable_refpoints_coords, all_refpoints

refpoints_file = f'{farmName} farm_reference_Points.csv'
unacceptable_refpoints_file = f'{farmName}-nanReferencePoints.csv'

acceptable_refpoints, unacceptable_refpoints, all_refpoints = filter_acceptable_refpoints(refpoints_file, unacceptable_refpoints_file)

print("Acceptable Reference Points:")
print(acceptable_refpoints)

print("\nUnacceptable Reference Points:")
print(unacceptable_refpoints)

print("\nAll Reference Points:")
print(all_refpoints)
# %% Forward Operator and response function
from pygimli.frameworks import Modelling, Inversion
from pygimli.viewer.mpl import drawModel1D
import pygimli as pg

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
# amVec = np.arange(1, 7) * 0.5  # Geophilius 2.0 (2023) # uses for Original data

b = 1.0
bmVec = np.sqrt(amVec**2+b**2)

# model space
thk = np.ones(15) * 0.1
nLayer = len(thk) + 1

dataNearestMean_harmfit_on_IF = np.column_stack((np.array(refName, dtype='int'), meanNearestPoints_harmfit_on_IF))

# %% Initialize the DC Forward Modelling Operator
fop = VESRhoModelling(thk, am=amVec, an=bmVec, bm=bmVec, bn=amVec)
#%% Error Model
error = np.ones_like(amVec) * 0.03 # used in inversion
# error = np.ones_like(np.arange(1, numSpacing+1)) * 0.03 # used in inversion
error_replaced = np.copy(error)  # Create a copy of the original error array
# error_replaced[dataNearestMean_harmfit_on_IF == 1000] = 10000  # Replace specific error values with 10000
# make a copy of the mean-data to replace NaN values with 1000 
data_with_replacement = np.copy(dataNearestMean_harmfit_on_IF)

#%%  Error Model

# Create a new error vector with the same size as data_with_replacement[:, 3:8]
error_replaced_mean = np.ones_like(data_with_replacement[:, 3:8]) * 0.03
error_replaced_individual = np.ones_like(data_with_replacement[:, 3:8]) * 0.03

# Replace NaN values with 1000 in the mean-data
data_with_replacement[np.isnan(data_with_replacement)] = 1000

# Identify the indices where data is replaced with 1000
indices_to_replace_mean = np.where(data_with_replacement[:, 3:8] == 1000)
#indices_to_replace_individual = np.where(data_with_replacement[:, 3:8] == 1000)

# Set error to 10000 for replaced values in both vectors
error_replaced_mean[indices_to_replace_mean] = 10000
#error_replaced_individual[indices_to_replace_individual] = 10000

# %% Inversion (For Nearest Points to Reference-Point) test
       
harmfit_data = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)

# Load data from the 'harmfitted_isolation_forest_data' array
data = harmfit_data

# Extract the Easting and Northing columns from the data
Eutm, Nutm = data[:, 0], data[:, 1]

# Extract the resistivity values (Rho) for different spacings from the data
Rho1, Rho2, Rho3, Rho4, Rho5 = data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7]

# Initialize empty lists to store the inversion results
StmodelsMean = []   # Stores the mean inversion models for each reference point
chi2Vec_mean = []   # Stores the chi-square values for the mean inversion of each reference point
chi2Vec_indiv = []  # Stores the chi-square values for the individual inversions of each reference point
chi2_indiv = []     # Stores the individual inversion models for each reference point

num_valid_points_list = []  # List to store the number of valid points for each reference point
skipped_points_list = []    # List to store the skipped points for each reference point


# Inversion of individual nearest points
for j, Data in enumerate(dataNearestMean_harmfit_on_IF):
    array = data
    dist = np.sqrt((array[:, 0] - Data[1])**2 + (array[:, 1] - Data[2])**2)
    dist_index = np.argmin(dist)
    nearestArray = np.arange(dist_index - 5, dist_index + 5)
    mydata = data[nearestArray][:, 3:8]
    chi2Vec_indiv = []
    Stmodels = []
    skipped_points = []  # Create an empty list to keep track of skipped points
    
    
    # Create an error matrix for individual data
    error_replaced_individual = np.ones_like(mydata) * 0.03
    # Replace NaN values with 1000
    mydata[np.isnan(mydata)] = 1000
    # Identify the indices where data is replaced with 1000
    indices_to_replace_individual = np.where(mydata == 1000)
    # Set error to 10000 for replaced values in both vectors
    error_replaced_individual[indices_to_replace_individual] = 10000
    

    # Inversion of individual nearest points
    for i, indivData in enumerate(mydata):
        #print(indivData, error_replaced_individual[i,:])
        if not np.isnan(indivData).any():
            # Check if indivData contains NaN values
            inv_indiv = Inversion(fop=fop)  # Passing the fwd operator in the inversion
            inv_indiv.setRegularization(cType=1)  # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
            modelInd = inv_indiv.run(indivData, error_replaced_individual[i,:], lam=20, lambdaFactor=0.9, startModel=300, verbose=True)  # Stating model 100 ohm.m, lam: regularization
            Stmodels.append(modelInd)
            np.savetxt(f'{farmName}_InvResultIndividualPoints_{j}th_point.csv', Stmodels, delimiter=';', fmt='%s')
            chi2 = inv_indiv.inv.chi2()
            chi2Vec_indiv.append(chi2)
        else:
            skipped_points.append(nearestArray[i])  # Add the index of the skipped point to the list
    chi2_indiv.append(np.array(chi2Vec_indiv))
    
    #chi2_indiv_array = np.array(chi2_indiv)
    np.savetxt(f'{farmName}_chi2_indiv.csv', chi2_indiv, delimiter=';', fmt='%s')

    num_valid_points = len(nearestArray) - len(skipped_points)
    num_valid_points_list.append(num_valid_points)

    if skipped_points:
        print(f'Skipped inversion for points with indices: {skipped_points}')
        skipped_points_list.append(skipped_points)

    # inversion of mean of the nearest points
    Rho = np.array(Data[3:8])
    #print(Rho)
    inv_mean = Inversion(fop=fop) # passing the fwd operator in the inversion
    inv_mean.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
    
    # Create an error matrix for mean data
    error_replaced_mean = np.ones_like(Rho) * 0.03
    # Replace NaN values with 1000
    Rho[np.isnan(Rho)] = 1000
    # Identify the indices where data is replaced with 1000
    indices_to_replace_mean = np.where(Rho == 1000)
    # Set error to 10000 for replaced values in both vectors
    error_replaced_mean[indices_to_replace_mean] = 10000
    
    print(Rho, error_replaced_mean)
    
    modelMean = inv_mean.run(Rho, error_replaced_mean, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization
    invMeanResponse = inv_mean.response.array()
    StmodelsMean.append(modelMean)
    np.savetxt(f'{farmName}_InvResultMeanPoints.csv', StmodelsMean, delimiter=';', fmt='%s')
    np.savetxt(f'{farmName}_invMeanResponse_{j}th point.csv', invMeanResponse, delimiter=';', fmt='%s')
    chi2_m = inv_mean.inv.chi2()
    chi2Vec_mean.append(chi2_m)
    np.savetxt(f'{farmName}_chi2_mean.csv', chi2Vec_mean, delimiter=';', fmt='%s')


# Print the information about the number of valid points and skipped points for each reference point
for i, Data in enumerate(dataNearestMean_harmfit_on_IF):
    print(f'Reference point {int(Data[0])}:')
    print(f'Number of valid points for inversion: {num_valid_points_list[i]}')
    if skipped_points_list[i]:
        print(f'Skipped inversion for points with indices: {skipped_points_list[i]}')
    print()


# %%  plots for Nearest points to Reference Points
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors

# Inversion results for 'Mean' of closest points
## importing chi2 for the plots
chi2VecIndiv = pd.read_csv(f'{farmName}_chi2_indiv.csv', delimiter=';', header=None)
chi2VecMean = pd.read_csv(f'{farmName}_chi2_mean.csv', delimiter=';', header=None)
# Convert chi2VecIndiv DataFrame to a NumPy array
chi2Indiv = np.array(chi2VecIndiv)
# Convert chi2VecMean DataFrame to a NumPy array
chi2Mean = np.array(chi2VecMean)

# # Find the maximum row length in chi2_indiv
# max_row_length = max(len(row) for row in chi2Indiv)

# # Create a new n by n array filled with zeros
# n_by_n_chi2_indiv = np.zeros((len(chi2_indiv), max_row_length))

# # Fill the new array with rows from chi2_indiv
# for i, row in enumerate(chi2_indiv):
#     n_by_n_chi2_indiv[i, :len(row)] = row


# # Save the new array to a CSV file
# np.savetxt(f'{farmName}_chi2_indiv_NbyN.csv', n_by_n_chi2_indiv, delimiter=';', fmt='%.4f')


# Load data from the 'harmfitted_isolation_forest_data' array
harmfit_data = np.genfromtxt(f"Combined_harmfit_array_{farmName}.txt", delimiter='\t', skip_header=1)
data = harmfit_data

with PdfPages(f'{farmName} Inversion_Result_ Mean&Indiv.pdf') as pdf:
    for j, Data in enumerate(dataNearestMean_harmfit_on_IF):
        #print(Data)
        # plot individual data
        array = data
        dist = np.sqrt((array[:,0]-Data[1])**2+(array[:,1]-Data[2])**2)
        
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index-5, dist_index+5)
        mydata = data[nearestArray][:, 3:8]  # indiv

        Rho = np.array(Data[3:8]) # mean
        # Replace NaN values with 1000
        #Rho[np.isnan(Rho)] = 1000  

        Inv_indiv = np.loadtxt(f'{farmName}_InvResultIndividualPoints_{j}th_point.csv' , delimiter=';')
        #print(Inv_indiv.astype(int).shape)

        #print(Inv_indiv.shape)
        Inv_mean = np.loadtxt(f'{farmName}_InvResultMeanPoints.csv', delimiter=';')
        InvMeanResponse = np.loadtxt(f'{farmName}_invMeanResponse_{j}th point.csv', delimiter=';')
        inv_mean = Inversion(fop=fop) # passing the fwd operator in the inversion
        
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15.5, 8), gridspec_kw={'height_ratios': [1, 1]})
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        ax0 = axes[0, 0]
        ax1 = axes[0, 1]
        ax2 = axes[0, 2]
        ax3 = axes[1, 0]
        ax4 = axes[1, 1]
        ax5 = axes[1, 2]

        ax0.bar(range(len(chi2Indiv[j])), chi2Indiv[j], width=0.3, label=f"chi2 for individual data {j}")
        ax0.axhline(chi2Mean[j], linestyle='--', c='r', label="chi2 for mean data")
        ax0.grid(True)
        ax0.set_xlim([0, len(chi2VecIndiv)])
        ax0.legend(fontsize=8, loc=4)
        ax0.set_xlabel('individual data')
        ax0.set_ylabel('$\u03C7^2$')
        ax0.set_title( f'Chi-Square')
        
        ax1.semilogx(Rho, amVec, "+", markersize=9, mew=2, label="Mean Data")
        ax1.semilogx(InvMeanResponse, amVec, "x", mew=2, markersize=8, label="Mean Response")
        ax1.invert_yaxis()
        ax1.set_xlim([5, 3000])
        ax1.grid(True) 
        ax1.semilogx(data[nearestArray][0, 3:8], amVec, ".", markersize=2, label="Individual Data")
        for i in range(1, mydata.shape[0]):
            ax1.semilogx(mydata[i, :], amVec, ".", markersize=2)
        ax1.legend(fontsize=8, loc=2)
        # ax1.set_title(f' Reference Point {Data[0]:.0f} - {farmName}',  loc='center', fontsize= 20)
        ax1.set_ylabel('spacing')
        ax1.set_title(f'Rhoa')
    
        
        drawModel1D(ax2, thickness=thk, values=Inv_mean[j], plot='semilogx', color='g', zorder=20, label="mean")
        ax2.set_xlim([5, 50000])
        ax2.legend(fontsize=8, loc=2)
        ax2.set_title('Model')
        for inv in Inv_indiv: 
            if isinstance(inv, np.ndarray) and inv.shape[0] > 0:  # Check if inv is a non-empty NumPy array
                drawModel1D(ax2, thickness=thk, values=inv, plot='semilogx', color='lightgray', linewidth=1)
        
        ax3.plot(Eutm[nearestArray], Nutm[nearestArray], "x", markersize=8, mew=2, label='Nearest Points')
        ax3.plot(all_refpoints['E'].iloc[j], all_refpoints['N'].iloc[j], "o", markersize=8, mew=2, label='Reference Point')
        ax3.axis("equal")
        ax3.set_title(f'Reference point {Data[0]:.0f} and its {len(nearestArray)} nearest points')
        ax3.legend(prop={'size': 8})
        ax3.set_xlabel('easting')
        ax3.set_ylabel('northing')
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.xaxis.set_tick_params(rotation=30)
        ax3.grid()

        matrixRho = np.vstack((mydata[:,0],mydata[:,1],mydata[:,2],mydata[:,3],mydata[:,4]))
        norm=colors.LogNorm(vmin=10, vmax=1000)
        mat=ax4.matshow(matrixRho, cmap='Spectral_r', norm=norm)
        ax4.axis("equal")
        ax4.set_title('Rhoa')
        ax4.set_xlabel(f'nearest data to the reference point {Data[0]:.0f} ')
        ax4.set_ylabel('spacing')
        ax4.set_ylim([4,0])
        ax4.set_ylim( auto=True)
        fig.colorbar(mat, ax=ax4, orientation='horizontal')
        


        # # Initialize an empty list to store individual models
        # Inv_indiv_list = []
        
        # # Loop through each iteration and create a list of individual models
        # for j in range(len(Inv_indiv)):
        #     model = Inv_indiv[j]
            
        #     if isinstance(model, np.ndarray) and model.ndim == 1:
        #         Inv_indiv_list.append(model.astype(int).tolist())
        #     elif np.isscalar(model):
        #         Inv_indiv_list.append([int(model)])
        #     else:
        #         print(f"Warning: Invalid individual model found for iteration {j}. Ignoring this iteration.")
        #         Inv_indiv_list.append([])  # Add an empty list for invalid models
        # #print(len(Inv_indiv_list[0]))

        # # Show the stitched models for one row (when we have just one non-Nan data for inversion)
        # if len(Inv_indiv_list[0]) == 1:
        #    # Draw the simple model using ax5.matshow
        #    matrixModel = np.array([Inv_indiv_list]).T
        #    norm = colors.LogNorm(vmin=10, vmax=1000)
        #    mod = ax5.matshow(matrixModel[-1, :, :], cmap='Spectral_r', norm=norm, aspect='auto') 
        #    ax5.set_title('Individual Model')
        #    ax5.set_xlabel('spacing')
        #    ax5.set_ylabel('model index')
        #    ax5.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        #    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        #    fig.colorbar(mod, ax=ax5, orientation='horizontal')
        # else:
        #    # Show the stitched models for multiple rows
        pg.viewer.mpl.showStitchedModels(Inv_indiv, ax=ax5, x=None, cMin=10, cMax=1000, cMap='Spectral_r', thk=thk, logScale=True, title='Model (Ohm.m)', zMin=0, zMax=0, zLog=False)
                
        
        # the main title
        fig.suptitle(f'{farmName} farm, Reference point {Data[0]:.0f}', fontsize=16)
        



        plt.savefig(pdf, format='pdf')
