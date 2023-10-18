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
import seaborn as sns
# %% import data

## Define file paths and farm names
# filepath = 'Geophilus_aufbereitet_2023-04-05_New.csv'
# farmName = 'aufbereitet'

# filepath = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName = 'Trebbin Wertheim'

filepath = 'Aussenschlag_2023_10_11_modified.csv'
farmName = 'Trebbin Aussenschlag'

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

# filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName = 'Boossen_1601'
#%% Functions
# filepath = filepath_original
# Function to read the processed data
def read_processed_data(filepath):
    # EdecDeg_P, NdecDeg_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, nan_data, Rho5_P, Gamma_P, BFI, lfd, Datum, Zeit = np.genfromtxt(
    #     filepath, skip_header=1, delimiter=',', unpack=True)
    EdecDeg_P, NdecDeg_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, nan_data, Rho5_P, Gamma_P, BFI, lfd, Datum, Zeit = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
    Eutm_P, Nutm_P, zone, letter = utm.from_latlon(NdecDeg_P, EdecDeg_P)
    farmName_Processed = 'TRB_Wertheim_Geophilus_roh_221125.csv'
    np.savetxt(f'{farmName_Processed} farm_data.csv', np.column_stack((Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P)), delimiter=';', fmt='%s')
    data_processed = np.column_stack((Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P))
    return data_processed

# Function to read the original data
def read_original_data(filepath):
    # EdecDeg_O, NdecDeg_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, nan_data, Rho5_O, Gamma_O = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
    EdecDeg_O, NdecDeg_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, nan_data, Rho5_O, Gamma_O, BFI, lfd, Datum, Zeit = np.genfromtxt(filepath, skip_header=1, delimiter=',', unpack=True)
    Eutm_O, Nutm_O, zone, letter = utm.from_latlon(NdecDeg_O, EdecDeg_O)
    farmName_original = 'TRB_Wertheim_Geophilus_roh_221125.csv'
    np.savetxt(f'{farmName_original}.txt', np.column_stack((Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O)), delimiter=';', fmt='%s')
    data_original = np.column_stack((Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O))
    num_spacing = data_original.shape[1]  # 5 spacing in Geophilus system

    return data_original


# Function to plot original data's areal plot
def plot_data(data_original, spacing_labels, farmName, refPoints, cmap='jet', vmin=10, vmax=1000):
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
        plt.scatter(refPoints['E'], refPoints['N'], c='red', marker='x', s=40, label='Reference Points')
        
        # Annotate each reference point with its corresponding index (number)
        for j, txt in enumerate(refPoints['Name']):
            plt.annotate(txt, (refPoints['E'][j], refPoints['N'][j]), fontsize=10, color='black')
        # Equalize the x and y axes
        plt.axis('equal')
        
        # colorbar
        plt.colorbar(sc)

        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Data - {spacing} - {farmName}', y=1.03)

        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()

        # Close the current figure
        plt.close()

    # Close the PDF file
    pdf_pages.close()



class LineDetector:
    def __init__(self, minPoints, maxAngleDeviation=20.):
        self.minPoints = minPoints   # Minimum number of points to consider a line

        self.maxAngleDeviation = maxAngleDeviation

    def detect_lines(self, rx, ry):
        """Split data into lines based on angle deviation and minimum points."""
        dummy = np.zeros_like(rx, dtype=int)
        line = np.zeros_like(rx, dtype=int)
        li = 0
        prev_angle = None

        for ri in range(1, len(rx)):
            dummy[ri - 1] = li

            # Calculate the angle between the current and previous points
            angle = np.arctan2(ry[ri] - ry[ri - 1], rx[ri] - rx[ri - 1]) * 180 / np.pi

            # Check if the angle deviation exceeds the threshold
            if prev_angle is not None and abs(angle - prev_angle) > self.maxAngleDeviation:
                li += 1

            prev_angle = angle
            dummy[ri] = li

        # Check if a line contains at least minPoints
        line_counts = np.bincount(dummy)
        for line_idx, point_count in enumerate(line_counts):
            if point_count < self.minPoints:
                line[dummy == line_idx] = -1  # Mark lines with less than minPoints as -1

        return self._sort_lines(rx, ry, line, dummy)

    def _sort_lines(self, rx, ry, line, dummy):
        """
        Sort line elements.
        """
        means = []
        for li in np.unique(dummy):
            means.append(np.mean(ry[dummy == li], axis=0))

        lsorted = np.argsort(means)
        for li, lold in enumerate(lsorted):
            line[dummy == lold] = li + 1

        return line





def plot_detected_lines(data_file, farmName, refPoints_df, maxDeviation, minPoints):
    # Read original data
    data_original = read_original_data(data_file)
    Eutm_O, Nutm_O = data_original[:, 0], data_original[:, 1]

    # Create LineDetector instance with desired parameters
    line_detector = LineDetector(minPoints, maxDeviation)

    # Detect lines
    line_indices = line_detector.detect_lines(Eutm_O, Nutm_O)

    # Get unique line indices and their counts
    unique_lines, counts = np.unique(line_indices, return_counts=True)

    # Filter lines with more than minPoints
    valid_lines = unique_lines[counts >= minPoints]
    
    # Store line numbers
    line_numbers = [i + 1 for i in range(len(valid_lines))]
    # Create a dictionary to map line indices to line_numbers
    line_number_mapping = dict(zip(valid_lines, line_numbers))

    # Find the maximum x and y coordinates among detected lines and reference points
    max_x = np.max(Eutm_O)
    max_y = np.max(Nutm_O)
    min_x = np.min(Eutm_O)
    min_y = np.min(Nutm_O)

    # Create a PDF file to save the plot
    pdf_filename = f"detected_lines_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    # Create a new figure at the beginning of the function
    plt.figure(figsize=(10, 8))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, line_num in enumerate(valid_lines):
        # Filter out lines with only 5 points
        if len(line_indices[line_indices == line_num]) > 5:
            # Plot lines for the valid_lines
            line_x = Eutm_O[line_indices == line_num]
            line_y = Nutm_O[line_indices == line_num]
            plt.plot(line_x, line_y, color=colors[i % len(colors)], label=f'Line {line_number_mapping[line_num]}')
            # Add line number annotation next to the line start
            plt.annotate(f' {line_number_mapping[line_num]}', (line_x[0], line_y[0]), textcoords="offset points", xytext=(10, 10),
                          ha='center', fontsize=10, color=colors[i % len(colors)])

    # Plot reference point locations
    plt.scatter(refPoints_df['E'], refPoints_df['N'], c='black', marker='x', s=40, label='Reference Points')
    for j, txt in enumerate(refPoints_df['Name']):
        plt.annotate(txt, (refPoints_df['E'][j], refPoints_df['N'][j]), fontsize=9, color='black', ha='right', va='bottom')

    # Set x and y axes with consistent maximum values
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Set plot labels and title
    plt.xlabel('Easting (UTM)')
    plt.ylabel('Northing (UTM)')
    plt.title(f'Detected Lines - {farmName}', y=1.03)

    # Equalize the x and y axes
    plt.axis('equal')

    # Add information about the closest line inside the plot (bottom left)
    bottom_text = []
    for j, refPoint in refPoints_df.iterrows():
        ref_x, ref_y = refPoint['E'], refPoint['N']
        closest_line = None
        min_distance = np.inf
        for line_num in valid_lines:
            line_x = Eutm_O[line_indices == line_num]
            line_y = Nutm_O[line_indices == line_num]
            distances = np.sqrt((line_x - ref_x) ** 2 + (line_y - ref_y) ** 2)
            min_dist = np.min(distances)
            if min_dist < min_distance:
                closest_line = line_num
                min_distance = min_dist
        closest_line_text = f'Ref. Point {refPoint["Name"]}: Closest Line - {line_number_mapping[closest_line]}'
        bottom_text.append(closest_line_text)

    # Join the closest line information and place it at the bottom left
    closest_line_info = '\n'.join(bottom_text)
    plt.annotate(closest_line_info, xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10, color='blue',
                 ha='left', va='bottom')
    
    # Save the plot as a JPEG image with the same name as the PDF
    jpeg_filename = f"detected_lines_{farmName}.jpeg"
    plt.savefig(jpeg_filename, format='jpeg', dpi=300)
    
    # Save the current plot to the PDF file
    pdf_pages.savefig()


    # Close the PDF file
    pdf_pages.close()





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
        plt.axis('equal')

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


def harmfit_without_nan(data, nc=30, error=0.01):
    # Filter out invalid data (nan values)
    valid_indices = np.isfinite(data[:, 2])
    valid_data = data[:, 2][valid_indices]

    if len(valid_data) == 0:
        print("No valid data points found. Skipping harmfit.")
        return [], [], []

    xline = np.sqrt((data[:, 0] - data[:, 0][0]) ** 2 + (data[:, 1] - data[:, 1][0]) ** 2)

    # Perform harmfit on the valid data
    harmfit_result, _ = harmfit(np.log(valid_data), x=xline[valid_indices], nc=nc, error=error)

    # Create a new array to store the harmfit result with nan values restored
    harmfit_result_with_nan = np.empty(len(data))
    harmfit_result_with_nan[:] = np.nan
    harmfit_result_with_nan[valid_indices] = np.exp(harmfit_result)

    x_harmfitted = data[:, 0].copy()
    y_harmfitted = data[:, 1].copy()

    return x_harmfitted[valid_indices], y_harmfitted[valid_indices], np.exp(harmfit_result)



def harmfit_on_IF(Isolation_Forest, spacing_labels, selected_lines, line_indices, farmName, refPoints_df, cmap='jet', vmin=10, vmax=1000):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    pdf_filename = f"Harmfit_Isolation_forest_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)
    nc = 30

    all_harmfitted_data = []  # List to store harmfitted data for all spacings

    # Iterate over each spacing
    for i, spacing in enumerate(spacing_labels):
        x_all = []
        y_all = []
        harmfitted_spacing_data_list_spacing = []  # List to store harmfitted data for the current spacing

        # Iterate over each selected line
        for line_num in selected_lines:
            point_indices_of_interest = np.where(line_indices == line_num)[0]
            point_indices_of_interest = point_indices_of_interest.astype(int)

            x_coords = Isolation_Forest[point_indices_of_interest, 0]
            y_coords = Isolation_Forest[point_indices_of_interest, 1]
            h = Isolation_Forest[point_indices_of_interest, 2]

            resistivity_spacing_line = Isolation_Forest[point_indices_of_interest, i + 3]

            x_harmfitted, y_harmfitted, harmfitted_IF_data = harmfit_without_nan(
                np.column_stack((x_coords, y_coords, resistivity_spacing_line)),
                nc=nc,
                error=0.01)

            x_all.extend(x_harmfitted)
            y_all.extend(y_harmfitted)
            harmfitted_spacing_data_list_spacing.extend(harmfitted_IF_data)

        # Append harmfitted data for the current spacing
        all_harmfitted_data.append(harmfitted_spacing_data_list_spacing)

        # Draw the areal plot for all lines together for the current spacing
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(x_all, y_all, s=0.7, c=harmfitted_spacing_data_list_spacing, cmap='jet', norm=LogNorm(vmin=10, vmax=1000))

        # Plot the reference points on the subplot
        plt.scatter(refPoints_df['E'], refPoints_df['N'], c='red', marker='x', s=30, label='Reference Points')

        # Annotate each reference point with its corresponding index (number)
        for j, txt in enumerate(refPoints_df['Name']):
            plt.annotate(txt, (refPoints_df['E'][j], refPoints_df['N'][j]), fontsize=9, color='black')

        # colorbar
        plt.colorbar(sc)

        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Harmfit on Isolation forest - {spacing} - {farmName}', y=1.03)
        plt.axis('equal')

        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()

        plt.close()

    # Close the PDF file
    pdf_pages.close()

    # Save the combined harmfitted data for all spacings in a text file. Since the sizes are dofferent
    harmfitted_data_file = f"harmfit_on_IF_{farmName}.txt"
    # Determine the maximum sizes
    max_size = max(len(x_all), len(y_all), len(max(all_harmfitted_data, key=len)))
    # Create empty arrays with the maximum size
    # x_combined = np.ones(max_size)
    # y_combined = np.ones(max_size)
    # z_combined = np.ones(max_size)

    x_combined = np.full(max_size, np.nan)
    y_combined = np.full(max_size, np.nan)
    z_combined = np.full(max_size, np.nan)
    num_harmonics = len(all_harmfitted_data)  
    harmonics_combined = np.full((max_size, num_harmonics), np.nan)
    # harmonics_combined = np.ones((max_size, len(all_harmfitted_data)))

    # Assign data from all_harmfitted_data to the corresponding arrays
    x_combined[:len(x_all)] = x_all
    y_combined[:len(y_all)] = y_all
    for i, data in enumerate(all_harmfitted_data):
        harmonics_combined[:len(data), i] = data
    # Stack the arrays
    data_to_save = np.column_stack((x_combined, y_combined, z_combined, harmonics_combined))

    np.savetxt(harmfitted_data_file, data_to_save, delimiter='\t')

    return  data_to_save 



def harmfit_on_org(original_data, spacing_labels, selected_lines, line_indices, farmName, refPoints_df, cmap='jet', vmin=10, vmax=1000):

    pdf_filename = f"Harmfit_original_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)
    nc = 30

    all_harmfitted_data = []  # List to store harmfitted data for all spacings

    # Iterate over each spacing
    for i, spacing in enumerate(spacing_labels):
        x_all = []
        y_all = []
        harmfitted_spacing_data_list_spacing = []  # List to store harmfitted data for the current spacing

        # Iterate over each selected line
        for line_num in selected_lines:
            point_indices_of_interest = np.where(line_indices == line_num)[0]
            point_indices_of_interest = point_indices_of_interest.astype(int)

            x_coords = original_data[point_indices_of_interest, 0]
            y_coords = original_data[point_indices_of_interest, 1]
            h = original_data[point_indices_of_interest, 2]

            resistivity_spacing_line = original_data[point_indices_of_interest, i + 3]

            x_harmfitted, y_harmfitted, harmfitted_org_data = harmfit_without_nan(
                np.column_stack((x_coords, y_coords, resistivity_spacing_line)),
                nc=nc,
                error=0.01)

            x_all.extend(x_harmfitted)
            y_all.extend(y_harmfitted)
            harmfitted_spacing_data_list_spacing.extend(harmfitted_org_data)

        # Append harmfitted data for the current spacing
        all_harmfitted_data.append(harmfitted_spacing_data_list_spacing)

        # Draw the areal plot for all lines together for the current spacing
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(x_all, y_all, s=0.7, c=harmfitted_spacing_data_list_spacing, cmap='jet', norm=LogNorm(vmin=10, vmax=1000))

        # Plot the reference points on the subplot
        plt.scatter(refPoints_df['E'], refPoints_df['N'], c='red', marker='x', s=30, label='Reference Points')

        # Annotate each reference point with its corresponding index (number)
        for j, txt in enumerate(refPoints_df['Name']):
            plt.annotate(txt, (refPoints_df['E'][j], refPoints_df['N'][j]), fontsize=9, color='black')

        # colorbar
        plt.colorbar(sc)

        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Harmfit on original - {spacing} - {farmName}', y=1.03)
        plt.axis('equal')

        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()

        plt.close()

    # Close the PDF file
    pdf_pages.close()

    # Save the combined harmfitted data for all spacings in a text file
    harmfitted_data_file = f"harmfit_on_original_{farmName}.txt"
    data_to_save = np.column_stack((x_all, y_all, y_all) + tuple(all_harmfitted_data))
    np.savetxt(harmfitted_data_file, data_to_save, delimiter='\t')

    return  data_to_save 





    

# Main function
def main():
    # data_file = "Beerfelde 2022_10_21-original.csv"
    # farmName = 'Beerfelde Fettke'
    
    # data_file = 'TRB_Wertheim_Geophilus_roh_221125.csv'
    # farmName = 'Trebbin Wertheim'
    data_file = 'Aussenschlag_2023_10_11_modified.csv'
    farmName = 'Trebbin Aussenschlag'
    
    spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

    # Read original data
    data_original = read_original_data(data_file)
    
    # Read processed data
    processed_data = read_processed_data(data_file)
    minPoints = 10.
    minDist = 10.
    maxAngleDeviation=10.
    # Create LineDetector instance with desired parameters
    #line_detector = LineDetector(minPoints, maxDeviation=22.)  # Adjust maxDeviation as needed
    line_detector = LineDetector(minPoints=5, maxAngleDeviation=20)

    # Detect lines
    line_indices = line_detector.detect_lines(data_original[:, 0], processed_data[:, 1])
    
    # Get unique line indices and their counts
    unique_lines, counts = np.unique(line_indices, return_counts=True)
    
    # Filter lines with more than 10 points
    selected_lines = unique_lines[counts >= 10]
    
    # Print detected lines with more than 10 points
    num_lines = len(selected_lines)
    num_unvalid_lines = len(unique_lines) - num_lines
    print("Number of valid lines detected: ", num_lines)
    print("Number of invalid lines (containing less than 10 points): ", num_unvalid_lines)
       
    
    # Import reference points from KML file
    # kmlFile = 'TRB_Wertheim_Referenzpunkte_gemessen_2022-12-01.kml'
    kmlFile = 'TRB_Außenschlag_Referenzpunkte_gemessen_2023-03-20.kml'
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    gdf = gpd.read_file(kmlFile)
    refLat = np.array(gdf['geometry'].y)
    refLon = np.array(gdf['geometry'].x)
    refEutm, refNutm, refzone, refletter = utm.from_latlon(refLat, refLon)
    refName = np.array(gdf['Name'])
    refPoints = np.column_stack([refName, refEutm, refNutm])
    header = ['Name', 'E', 'N']
    refPointsTable = np.vstack((header, refPoints))
    np.savetxt(f'{farmName}_farm_reference_Points.csv', refPointsTable, delimiter=';', fmt='%s')
    refPoints = pd.DataFrame(refPoints, columns=['Name', 'E', 'N'])

    # Call the plot_detected_lines function
    #plot_detected_lines(data_file, farmName, refPoints, maxDeviation, minPoints)
    plot_detected_lines(data_file, farmName, refPoints, maxAngleDeviation, minPoints)
    
    # Call the function to generate the separate PDF with detected lines
    # plot_lines_separately(data_file, farmName, maxAngleDeviation, refPoints, minPoints)
    
    # Plot original data's areal plot
    # plot_data(processed_data, spacing_labels, farmName, refPoints)
    
    # Perform Isolation Forest
    # num_spacing = len(spacing_labels)
    # Isolation_Forest = isolation_forest_outlier_detection(data_original, num_spacing, refPoints)
    
    # Perform harmfit on IF result and draw areal plots
    # harmfit_on_IF(Isolation_Forest, spacing_labels, selected_lines, line_indices, farmName, refPoints)
    
    # Perform harmfit on ORIGINAL DATA result and draw areal plots
    # harmfit_on_org(data_original, spacing_labels, selected_lines, line_indices, farmName, refPoints)
if __name__ == "__main__":
    main()

