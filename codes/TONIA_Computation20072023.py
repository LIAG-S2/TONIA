#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:10:53 2023

@author: sadegh
"""
# %% import lberaries
import numpy as np
import utm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pygimli.frameworks import harmfit
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import geopandas as gpd
import fiona
# %% import data

import numpy as np
import utm



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
farmName_Processed = 'Beerfelde Fettke Processed data'
filepath_original = 'Beerfelde 2022_10_21-original.csv'
farmName_original = 'Beerfelde Fettke Original data'

# filepath = 'BDW_Hoben_Geophilus_221020_roh.csv'
# farmName = 'Brodowin Hoben'

# filepath = 'BDW_Seeschlag_Geophilus_roh_221020 - Copy.csv'
# farmName = 'Brodowin Seeschlag'

# filepath = 'LPP_1211_Geophilus_170411_roh_EPSG04326.csv'
# farmName = 'Boossen_1211'

# filepath = 'LPP_1601_Geophilus_221019_roh_EPSG04326.csv'
# farmName = 'Boossen_1601'



def read_processed_data(filepath):
    EdecDeg_P, NdecDeg_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Rho5_P, Gamma_P, BFI, lfd, Datum, Zeit = np.genfromtxt(
        filepath, skip_header=1, delimiter=',', unpack=True)
    Eutm_P, Nutm_P, zone, letter = utm.from_latlon(NdecDeg_P, EdecDeg_P)
    farmName_Processed = 'Beerfelde Fettke Processed data'
    np.savetxt(f'{farmName_Processed} farm_data.csv', np.column_stack((Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P)), delimiter=';', fmt='%s')
    return Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P

def read_original_data(filepath):
    EdecDeg_O, NdecDeg_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Rho5_O, Gamma_O = np.genfromtxt(
        filepath, skip_header=1, delimiter=',', unpack=True)
    Eutm_O, Nutm_O, zone, letter = utm.from_latlon(NdecDeg_O, EdecDeg_O)
    farmName_original = 'Beerfelde Fettke Original data'
    np.savetxt(f'{farmName_original} farm_data.csv', np.column_stack((Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O)), delimiter=';', fmt='%s')
    return Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O



# Read processed and original data using the functions and unpack them
Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P = read_processed_data(filepath_Processed)
Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O = read_original_data(filepath_original)

# Check the shape of unpacked data
print("Processed data shape:", Eutm_P.shape, Nutm_P.shape, H_P.shape, Rho1_P.shape, Rho2_P.shape, Rho3_P.shape, Rho4_P.shape, Rho5_P.shape, Gamma_P.shape)
print("Original data shape:", Eutm_O.shape, Nutm_O.shape, H_O.shape, Rho1_O.shape, Rho2_O.shape, Rho3_O.shape, Rho4_O.shape, Rho5_O.shape, Gamma_O.shape)


#%% Line Detection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
        Sort line elements by Rx or Ry coordinates.
        """
        means = []
        for li in np.unique(dummy):
            means.append(np.mean(ry[dummy==li], axis=0))

        lsorted = np.argsort(means)
        for li, lold in enumerate(lsorted):
            line[dummy == lold] = li + 1

        return line
    
    # x and y coordinates
x_coordinates = Eutm_O
y_coordinates = Nutm_O

# Create LineDetector instance with desired parameters
line_detector = LineDetector(minDist=200, maxDeviation=155)

# Detect lines
line_indices = line_detector.detect_lines(x_coordinates, y_coordinates)

# Get unique line indices and their counts
unique_lines, counts = np.unique(line_indices, return_counts=True)

# Filter lines with more than 10 points
selected_lines =  unique_lines[counts >= 10]

# Print detected lines with more than 10 points
num_lines = len(selected_lines)
num_unvalid_lines = len(unique_lines) - len(selected_lines)
print("Number of valid lines detected: ", num_lines)
print("Number of unvalid lines (containing less than 10 points): ", num_unvalid_lines)


pdf_filename_lines = f"line_plots {farmName}.pdf"
pdf_pages_lines = PdfPages(pdf_filename_lines)

# Plot valid lines
plt.figure(figsize=(8, 6))

for line_idx, line_num in enumerate(selected_lines, start=1):
    indices = np.where(line_indices == line_num)[0]
    x_line = x_coordinates[indices]
    y_line = y_coordinates[indices]
    plt.plot(x_line, y_line)
    
    # Write line index as label at the end of the line
    label_x = x_line[-1]
    label_y = y_line[-1]
    plt.text(label_x, label_y, str(line_idx), fontsize=10, ha='center', va='center')

plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title(f'Line Plot - {farmName}')
plt.grid(True)

# Save the figure to the PDF file
pdf_pages_lines.savefig()

# Close the PDF file
pdf_pages_lines.close()

#%% Harmfit Outlier Detection Calculation
from pygimli.frameworks import harmfit

def harmfit_outlier_detection(Eutm, Nutm, H, Rho_original, Gamma, line_indices, selected_lines, spacing_labels, colors_, farmName):
    # Create a PDF file to save the plots
    pdf_filename = f"harmfit_results_lines - {farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    # Save the harmfitted data in a text file
    harmfit_filename = f"harmfit_data - {farmName}.txt"
    harmfit_file = open(harmfit_filename, 'w')
    harmfit_file.write("X-coordinate\tY-coordinate\tH\tSpacing 1\tSpacing 2\tSpacing 3\tSpacing 4\tSpacing 5\tGamma\n")
    # Iterate over each line
    for line_num in selected_lines:
        point_indices_of_interest = np.where(line_indices == line_num)[0]

        # Initialize the empty array to store the harmfitted data
        num_points = len(point_indices_of_interest)
        harmfitted_data = np.zeros((num_points, 9))

        for i in range(Rho_original.shape[1]):  # Iterate over columns of Rho_original
            rho = Rho_original[:, i]  # Get the relevant column of Rho_original
            smoothed_Rho_harmfit = np.exp(harmfit(np.log(rho[point_indices_of_interest]), Eutm[point_indices_of_interest], nc=50, error=0.1)[0])

            # Make sure smoothed_Rho_harmfit has shape (num_points, 1)
            smoothed_Rho_harmfit = smoothed_Rho_harmfit.reshape(-1, 1)

            # Save the required data in the text file
            harmfitted_data[:, 0] = Eutm[point_indices_of_interest]
            harmfitted_data[:, 1] = Nutm[point_indices_of_interest]
            harmfitted_data[:, 2] = H[point_indices_of_interest]
            harmfitted_data[:, i + 3] = smoothed_Rho_harmfit[:, 0]
            harmfitted_data[:, 8] = Gamma[point_indices_of_interest]

        np.savetxt(harmfit_file, harmfitted_data, delimiter='\t', fmt='%.6f')

        # plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        # plot harmfitted
        for j, spacing in enumerate(spacing_labels):
            ax.plot(
                Eutm[point_indices_of_interest],
                harmfitted_data[:, j + 3],
                label=f'harmfit Data - {spacing}',
                color=colors_[j]
            )
        # plot original data
        for j, rho in enumerate(Rho_original.T):
            ax.scatter(
                Eutm[point_indices_of_interest],
                rho[point_indices_of_interest],
                label=f'Original Data - {spacing_labels[j]}',
                color=colors_[j],
                marker='o',
                s=10
            )

        ax.set_xlabel('Easting (UTM)')
        ax.set_ylabel('Resistivity')
        ax.set_title(f'Line {line_num} - {farmName}')
        ax.set_yscale('log')
        ax.legend()

        plt.tight_layout()
        plt.show()

        pdf_pages.savefig(fig)

    harmfit_file.close()
    pdf_pages.close()

# Read processed and original data using the functions and unpack them
Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P = read_processed_data(filepath_Processed)
Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O = read_original_data(filepath_original)

# Detect lines
line_detector = LineDetector(minDist=200, maxDeviation=155)
line_indices = line_detector.detect_lines(Eutm_O, Nutm_O)

# Get unique line indices and their counts
unique_lines, counts = np.unique(line_indices, return_counts=True)

# Filter lines with more than 10 points
selected_lines = unique_lines[counts >= 10]

# Define spacing labels and colors
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']
colors_ = ['red', 'blue', 'green', 'orange', 'purple']

Rho_original = np.column_stack((Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O))

# Perform harmfit outlier detection and plotting
harmfit_outlier_detection(Eutm_O, Nutm_O, H_O, Rho_original, Gamma_O, line_indices, selected_lines, spacing_labels, colors_, farmName)

#%%  Areal plot for harmfit data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import utm


def draw_areal_plot(data_file, spacing_labels, cmap='jet', vmin=10, vmax=1000):
    # Read the harmfitted data from the txt file
    harmfitted_data = np.loadtxt(data_file, delimiter='\t', skiprows=1)

    # Get the UTM coordinates, resistivity values, and spacing names
    Eutm = harmfitted_data[:, 0]
    Nutm = harmfitted_data[:, 1]
    resistivity_harmfit = harmfitted_data[:, 3:8]

    # Create a multi-page PDF file to save the plots
    pdf_filename = f"Harmfit areal_plots_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    # Iterate over each spacing and plot on a separate page
    for i, spacing in enumerate(spacing_labels):
        # Create a new figure for each spacing plot
        plt.figure(figsize=(8, 6))

        # Get the resistivity values for the current spacing
        resistivity_spacing = resistivity_harmfit[:, i]

        # Draw the areal plot for the current spacing
        sc = plt.scatter(Eutm, Nutm, s=0.7, c=resistivity_spacing, cmap=cmap, norm=LogNorm(vmin=10, vmax=1000))

        # Add colorbar
        plt.colorbar(sc)

        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Harmfit Areal Plot - {spacing} - {farmName}')

        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()

        # Close the current figure
        plt.close()

    # Close the PDF file
    pdf_pages.close()

# Draw areal plot using the saved TXT file for each spacing
data_file = f"harmfit_data - {farmName}.txt"
draw_areal_plot(data_file, spacing_labels)





# %% IsolationForest algorithm outlier detection

import numpy as np
from sklearn.ensemble import IsolationForest
# Set a fixed random seed
np.random.seed(42)
# Resistivity data and (X,Y) UTM coordinates
Rho_original = np.column_stack((Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O))
x_original = Eutm_O
y_original = Nutm_O
H_original = H_O
Gamma_original = Gamma_O

num_spacing = Rho_original.shape[1]  # 5 spacing in Geophilus system

def isolation_forest_outlier_detection(resistivity_data, x_coords, y_coords, num_spacing):
    # Create an array to store the Isolation Forest results
    resistivity_Isolation_Forest = np.empty((len(resistivity_data), num_spacing + 4))
    resistivity_Isolation_Forest[:] = np.nan

    # Store the (X,Y) UTM coordinates, H_original, and Gamma_original
    resistivity_Isolation_Forest[:, 0] = x_coords
    resistivity_Isolation_Forest[:, 1] = y_coords
    resistivity_Isolation_Forest[:, 2] = H_original
    resistivity_Isolation_Forest[:, -1] = Gamma_original

    num_outliers = np.zeros(num_spacing)  # Array to store the number of outliers per spacing

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
        resistivity_Isolation_Forest[outliers_iforest == 1, spacing + 3] = selected_values

    # Save the Isolation Forest data to a single text file
    output_filename = f"Isolation_{farmName.replace(' ', '')}.txt"
    header = 'x\ty\th\t' + '\t'.join([f"Rho Spacing {i+1}" for i in range(num_spacing)]) + '\tGamma'
    np.savetxt(output_filename, resistivity_Isolation_Forest, delimiter='\t', header=header)

    # Print the number of outliers and non-outliers per spacing
    for spacing in range(num_spacing):
        print(f"Spacing {spacing + 1}: Outliers = {int(num_outliers[spacing])}, Non-Outliers = {int(len(resistivity_data) - num_outliers[spacing])}")

# Call the function with the original resistivity data, X and Y UTM coordinates, and the number of spacings
isolation_forest_outlier_detection(Rho_original, x_original, y_original, num_spacing)






#%% IsolationForest Areal plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

# Define spacing labels
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

# Load the Isolation Forest data from the saved TXT file
data_file = f"Isolation_{farmName.replace(' ', '')}.txt"
isolation_forest_data = np.loadtxt(data_file, delimiter='\t', skiprows=1)

# Get the (X,Y) UTM coordinates
x_spacing = isolation_forest_data[:, 0]
y_spacing = isolation_forest_data[:, 1]

# Create a multi-page PDF file to save the plots
pdf_filename = f"Isolation_Forest_areal_plots_{farmName.replace(' ', '')}.pdf"
pdf_pages = PdfPages(pdf_filename)

# Iterate over each spacing and plot on a separate page
for i, spacing in enumerate(spacing_labels):
    # Get the resistivity values for the current spacing
    resistivity_spacing = isolation_forest_data[:, i + 3]

    # Create a new figure for each spacing plot
    plt.figure(figsize=(8, 6))

    # Draw the areal plot for the current spacing
    sc = plt.scatter(x_spacing, y_spacing, s=0.7, c=resistivity_spacing, cmap='jet', norm=LogNorm(vmin=10, vmax=1000))

    # Add colorbar
    plt.colorbar(sc)

    # Set plot labels and title
    plt.xlabel('Easting (UTM)')
    plt.ylabel('Northing (UTM)')
    plt.title(f'Isolation Forest Areal Plot - {spacing} - {farmName}')

    # Save the current plot to the PDF file
    pdf_pages.savefig()

    # Show the current plot on the screen
    plt.show()

    # Close the current figure
    plt.close()

# Close the PDF file
pdf_pages.close()

#%%  import reference points from kml file
import geopandas as gpd
import fiona
import pandas as pd

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

#%% Harmfit on Isolation_Forest (calculation & plot)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

def harmfit_without_nan(data, nc=50, error=0.01):
    # Filter out invalid data (nan values)
    valid_indices = np.isfinite(data)
    valid_data = data[valid_indices]

    # Perform harmfit on the valid data
    harmfit_result, _ = harmfit(np.log(valid_data), nc=nc, error=error)

    # Create a new array to store the harmfit result with nan values restored
    harmfit_result_with_nan = np.empty(len(data))
    harmfit_result_with_nan[:] = np.nan
    harmfit_result_with_nan[valid_indices] = np.exp(harmfit_result)

    return harmfit_result_with_nan

def draw_harmfit_areal_plots_on_IF(data_file, spacing_labels, farmName, cmap='jet', vmin=10, vmax=1000):
    # Read the Isolation Forest data from the txt file
    isolation_forest_data = np.loadtxt(data_file, delimiter='\t', skiprows=1)

    # Get the (X,Y) UTM coordinates
    x_spacing = isolation_forest_data[:, 0]
    y_spacing = isolation_forest_data[:, 1]
    h_spacing = isolation_forest_data[:, 2]

    # Create a multi-page PDF file to save the plots
    pdf_filename = f"Harmfit_Isolation_forest_{farmName}.pdf"
    pdf_pages = PdfPages(pdf_filename)

    harmfitted_data_list = []  # List to store harmfitted data for each spacing

    # Iterate over each spacing and plot on a separate page
    for i, spacing in enumerate(spacing_labels):
        # Get the resistivity values for the current spacing
        resistivity_spacing = isolation_forest_data[:, i + 3]

        # Perform harmfit on the Isolation Forest resistivity data
        harmfitted_data = harmfit_without_nan(resistivity_spacing, nc=160, error=0.1)
        harmfitted_data_list.append(harmfitted_data)  # Append harmfitted data to the list

        # Create a new figure for each spacing plot
        plt.figure(figsize=(8, 6))

        # Draw the areal plot for the harmfitted data points
        sc = plt.scatter(x_spacing, y_spacing, s=0.7, c=harmfitted_data, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        # Plot the reference points on the 5th subplot
        plt.scatter(refPoints_df['E'], refPoints_df['N'], c='red', marker='x', s=50, label='Reference Points')
        # Annotate each reference point with its corresponding index (number)
        for i, txt in enumerate(refPoints_df['Name']):
            plt.annotate(txt, (refPoints_df['E'][i], refPoints_df['N'][i]), fontsize=13, color='black')

        # Add colorbar
        plt.colorbar(sc)

        # Set plot labels and title
        plt.xlabel('Easting (UTM)')
        plt.ylabel('Northing (UTM)')
        plt.title(f'Harmfit on Isolation forest - {spacing} - {farmName}', y=1.03)

        # Save the current plot to the PDF file
        pdf_pages.savefig()
        plt.show()

        # Close the current figure
        plt.close()

    # Close the PDF file
    pdf_pages.close()

    # Save the harmfitted data for each spacing in a new column in the text file
    headers = 'X-coordinate\tY-coordinate\tH\t' + '\t'.join([f'{spacing}' for spacing in spacing_labels]) + '\tGamma'
    harmfitted_data_combined = np.column_stack([x_spacing, y_spacing, h_spacing] + harmfitted_data_list + [isolation_forest_data[:, -1]])
    harmfitted_on_IF_file = f"harmfit_on_IF_{farmName.replace(' ', '')}.txt"
    np.savetxt(harmfitted_on_IF_file, harmfitted_data_combined, delimiter='\t', header=headers)

    return harmfitted_data_list

data_file = f"Isolation_{farmName.replace(' ', '')}.txt"
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

# Draw harmfit areal plots using the saved TXT file for each spacing and get harmfitted data
harmfitted_data_list = draw_harmfit_areal_plots_on_IF(data_file, spacing_labels, farmName)

#%% #%% Draw subplot for 5 areal plots together and save them in a multi-layered PDF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

# Load Original and processed data
original_data = np.vstack((Eutm_O, Nutm_O, H_O, Rho1_O, Rho2_O, Rho3_O, Rho4_O, Rho5_O, Gamma_O)).T
processed_data = np.vstack((Eutm_P, Nutm_P, H_P, Rho1_P, Rho2_P, Rho3_P, Rho4_P, Rho5_P, Gamma_P)).T

# Load the Isolation Forest data from the saved TXT file
isolation_forest_data = np.genfromtxt(f"Isolation_{farmName.replace(' ', '')}.txt", delimiter='\t', skip_header=1)

# Load the harmfitted data from the saved TXT file
harmfit_data = np.genfromtxt(f"harmfit_data - {farmName}.txt", delimiter='\t', skip_header=1)

# Load the harmfitted data on Isolation Forest from the saved TXT file
harmfitted_isolation_forest_data = np.genfromtxt(f"harmfit_on_IF_{farmName.replace(' ', '')}.txt", delimiter='\t', skip_header=1)

# Define spacing labels
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

# Create a multi-page PDF file to save the plots
pdf_filename = f"subplot_plots_{farmName}.pdf"
pdf_pages = PdfPages(pdf_filename)

# Iterate over each spacing and create a subplot with 1 row and 5 columns
for i, spacing in enumerate(spacing_labels):
    # Create a new figure for the subplot
    plt.figure(figsize=(20, 4))

    # Original Data areal plot
    plt.subplot(1, 5, 1)
    plt.scatter(original_data[:, 0], original_data[:, 1], s=0.7, c=original_data[:, i + 3], cmap='jet', norm=LogNorm(vmin=10, vmax=1000))
    plt.axis('equal')
    plt.xlabel('Easting (UTM)')
    plt.ylabel('Northing (UTM)')
    plt.title('Original Data', y=1.03)
    plt.xticks(rotation=45)

    # Processed Data areal plot
    plt.subplot(1, 5, 2)
    plt.scatter(processed_data[:, 0], processed_data[:, 1], s=0.7, c=processed_data[:, i + 3], cmap='jet', norm=LogNorm(vmin=10, vmax=1000))
    plt.axis('equal')
    plt.xlabel('Easting (UTM)')
    #plt.ylabel('Northing (UTM)')
    plt.title('Processed Data', y=1.03)
    plt.xticks(rotation=45)

    # Harmfitted Data areal plot
    plt.subplot(1, 5, 3)
    plt.scatter(harmfit_data[:, 0], harmfit_data[:, 1], s=0.7, c=harmfit_data[:, i + 3], cmap='jet', norm=LogNorm(vmin=10, vmax=1000))
    plt.axis('equal')
    plt.xlabel('Easting (UTM)')
    #plt.ylabel('Northing (UTM)')
    plt.title('Harmfitted Data', y=1.03)
    plt.xticks(rotation=45)

    # Isolation Forested Data areal plot
    plt.subplot(1, 5, 4)
    plt.scatter(isolation_forest_data[:, 0], isolation_forest_data[:, 1], s=0.7, c=isolation_forest_data[:, i + 3], cmap='jet', norm=LogNorm(vmin=10, vmax=1000))
    plt.axis('equal')
    plt.xlabel('Easting (UTM)')
    #plt.ylabel('Northing (UTM)')
    plt.title('Isolation Forested Data', y=1.03)
    plt.xticks(rotation=45)


    # Harmfitted of Isolation Forested Data areal plot
    plt.subplot(1, 5, 5)
    plt.scatter(harmfitted_isolation_forest_data[:, 0], harmfitted_isolation_forest_data[:, 1], s=0.7, c=harmfitted_isolation_forest_data[:, i + 3], cmap='jet', norm=LogNorm(vmin=10, vmax=1000))
    plt.axis('equal')  # Set same axis limits as other subplots

    plt.xlabel('Easting (UTM)')
    #plt.ylabel('Northing (UTM)')
    plt.title('Harmfitted of Isolation Forest Data', y=1.03)
    plt.colorbar(label='Resistivity', orientation='vertical')

    # Adjust the spacing between subplots
    plt.tight_layout(w_pad=3.5)
    plt.xticks(rotation=45)
    
    plt.suptitle(f"{spacing} - {farmName}", fontsize=18, y=1.06)
    
    # Save the current plot to the PDF file
    pdf_pages.savefig(bbox_inches='tight')

# Close the PDF file
pdf_pages.close()

# Show the plots on the screen
plt.show()


# %% Find nearest dataPoints to refPoints

import numpy as np
import pandas as pd

def find_nearest_points(farmName, harmfitted_isolation_forest_data):
    refpoints = f'{farmName} farm_reference_Points.csv'
    RefPoints = pd.read_csv(refpoints, delimiter=";")

    nearest_points_list = []

    for point in RefPoints.itertuples():
        array = harmfitted_isolation_forest_data
        dist = np.sqrt((array[:, 0] - point[2]) ** 2 + (array[:, 1] - point[3]) ** 2)
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index - 5, dist_index + 5)

        data_nearest = array[nearestArray]   # data of individual closest points
        nearest_points_list.append(data_nearest)

    return nearest_points_list, RefPoints

def calculate_mean_nearest_points(nearest_points_list):
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

#  usage
harmfitted_isolation_forest_data = np.genfromtxt(f"harmfit_on_IF_{farmName.replace(' ', '')}.txt", delimiter='\t', skip_header=1)

nearest_points_list, refPoints = find_nearest_points(farmName, harmfitted_isolation_forest_data)
mean_nearest_points, nan_reference_points = calculate_mean_nearest_points(nearest_points_list)
mean_nearest_points_array = mean_nearest_points.reshape(-1, mean_nearest_points.shape[-1])

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
np.savetxt(f'{farmName}-meanNearestPoints-noNaN.csv', mean_nearest_points_array[~np.any(np.isnan(mean_nearest_points_array), axis=1)], delimiter=';', header=';'.join(header), fmt='%.6f')

# Save the original mean data of nearest points to another CSV file with NaN rows
np.savetxt(f'{farmName}-meanNearestPoints-withNaN.csv', mean_nearest_points_array, delimiter=';', header=';'.join(header), fmt='%.6f')





# %% import data for inversion
import pandas as pd

refPoints = pd.read_csv(f'{farmName} farm_reference_Points.csv', delimiter=';')
refName = refPoints_cleaned['Name'].astype(int) # refPoint names
meanNearestPoints_harmfit_on_IF = np.loadtxt(f'{farmName}-meanNearestPoints-noNaN.csv', delimiter=';') 
#meanNearestPoints_processed = np.loadtxt(f'{farmName}-meanNearestPoints_processed.csv', delimiter=';')
#data_O = np.loadtxt(f'{farmName_original} farm_data.csv', delimiter=';')
data_harmfit = np.loadtxt(f"harmfit_data - {farmName}.txt", delimiter='\t', skiprows=1)
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

    return acceptable_refpoints, unacceptable_refpoints_coords

refpoints_file = f'{farmName} farm_reference_Points.csv'
unacceptable_refpoints_file = f'{farmName}-nanReferencePoints.csv'

acceptable_refpoints, unacceptable_refpoints = filter_acceptable_refpoints(refpoints_file, unacceptable_refpoints_file)

print("Acceptable Reference Points:")
print(acceptable_refpoints)

print("\nUnacceptable Reference Points:")
print(unacceptable_refpoints)

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

dataNearestMean_harmfit_on_IF = np.column_stack((np.array(refName, dtype = 'int'), meanNearestPoints_harmfit_on_IF))

# %% Initialize the DC Forward Modelling Operator
fop = VESRhoModelling(thk, am=amVec, an=bmVec, bm=bmVec, bn=amVec)

#  Error Model
error = np.ones_like(amVec) * 0.03 # used in inversion
#error = np.ones_like(np.arange(1, numSpacing+1)) * 0.03 # used in inversion


# %% Inversion (For Nearest Points to Reference-Point)
       
harmfitted_isolation_forest_data = np.genfromtxt(f"harmfit_on_IF_{farmName.replace(' ', '')}.txt", delimiter='\t', skip_header=1)

# Load data from the 'harmfitted_isolation_forest_data' array
data = harmfitted_isolation_forest_data

# Extract the Easting and Northing columns from the data
Eutm, Nutm = data[:, 0], data[:, 1]

# Extract the resistivity values (Rho) for different layers from the data
Rho1, Rho2, Rho3, Rho4, Rho5 = data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]

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
    nearestArray = np.arange(dist_index - 6, dist_index + 6)
    mydata = data[nearestArray][:, 3:8]
    chi2Vec_indiv = []
    Stmodels = []
    skipped_points = []  # Create an empty list to keep track of skipped points

    # Inversion of individual nearest points
    for i, indivData in enumerate(mydata):
        if not np.isnan(indivData).any():  # Check if indivData contains NaN values
            inv_indiv = Inversion(fop=fop)  # Passing the fwd operator in the inversion
            inv_indiv.setRegularization(cType=1)  # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
            modelInd = inv_indiv.run(indivData, error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True)  # Stating model 100 ohm.m, lam: regularization
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
    inv_mean = Inversion(fop=fop) # passing the fwd operator in the inversion
    inv_mean.setRegularization(cType=1) # cType=0:MarquardtLevenberg damping, 10:mixe M.L. & smoothness Constraint
    modelMean = inv_mean.run(Rho, error, lam=20, lambdaFactor=0.9, startModel=300, verbose=True) # stating model 100 ohm.m, lam: regularization
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
chi2Indiv = np.array(chi2_indiv)

# Convert chi2VecMean DataFrame to a NumPy array
chi2Mean = chi2VecMean.to_numpy()

# Find the maximum row length in chi2_indiv
max_row_length = max(len(row) for row in chi2Indiv)

# Create a new n by n array filled with zeros
n_by_n_chi2_indiv = np.zeros((len(chi2_indiv), max_row_length))

# Fill the new array with rows from chi2_indiv
for i, row in enumerate(chi2_indiv):
    n_by_n_chi2_indiv[i, :len(row)] = row

# Save the new array to a CSV file
np.savetxt(f'{farmName}_chi2_indiv_NbyN.csv', n_by_n_chi2_indiv, delimiter=';', fmt='%.4f')



with PdfPages(f'{farmName} Inversion_Result_ Mean&Indiv.pdf') as pdf:
    for j, Data in enumerate(dataNearestMean_harmfit_on_IF):
        # plot individual data
        array = data
        dist = np.sqrt((array[:,0]-Data[1])**2+(array[:,1]-Data[2])**2)
        
        dist_index = np.argmin(dist)
        nearestArray = np.arange(dist_index-6, dist_index+6)
        Rho = np.array(Data[3:8])
        mydata = data[nearestArray][:, 3:8]


        Inv_indiv = np.loadtxt(f'{farmName}_InvResultIndividualPoints_{j}th_point.csv' , delimiter=';')
        print(Inv_indiv.astype(int).shape)

        #print(Inv_indiv.shape)
        Inv_mean = np.loadtxt(f'{farmName}_InvResultMeanPoints.csv', delimiter=';')
        InvMeanResponse = np.loadtxt(f'{farmName}_invMeanResponse_{j}th point.csv', delimiter=';')
        inv_mean = Inversion(fop=fop) # passing the fwd operator in the inversion
        
        fig = plt.figure(figsize=(15.5, 8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        spec = mpl.gridspec.GridSpec(ncols=3, nrows=2)
        ax0 = fig.add_subplot(spec[0,0])
        ax1 = fig.add_subplot(spec[0,1])
        ax2 = fig.add_subplot(spec[0,2])
        ax3 = fig.add_subplot(spec[1,0])
        ax4 = fig.add_subplot(spec[1,1])        
        ax5 = fig.add_subplot(spec[1,2])

        ax0.bar(range(len(n_by_n_chi2_indiv[j])), n_by_n_chi2_indiv[j], width=0.3, label=f"chi2 for individual data {j}")
        ax0.axhline(chi2Mean[j], linestyle='--', c='r', label="chi2 for mean data")
        ax0.grid(True)
        ax0.set_xlim([0, len(chi2VecIndiv)])
        ax0.legend(fontsize=8, loc=4)
        ax0.set_xlabel('individual data')
        ax0.set_ylabel('$\u03C7^2$')
        ax0.set_title( f'Chi-Square')
        
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
        ax2.set_xlim([5, 50000])
        ax2.legend(fontsize=8, loc=2)
        ax2.set_title('Model')
        for inv in Inv_indiv: 
            if isinstance(inv, np.ndarray) and inv.shape[0] > 0:  # Check if inv is a non-empty NumPy array
                drawModel1D(ax2, thickness=thk, values=inv, plot='semilogx', color='lightgray', linewidth=1)
        
        ax3.plot(Eutm[nearestArray], Nutm[nearestArray], "x", markersize=8, mew=2, label='Nearest Points')
        ax3.plot(acceptable_refpoints['E'].iloc[j], acceptable_refpoints['N'].iloc[j], "o", markersize=8, mew=2, label='Reference Point')
        ax3.axis("equal")
        ax3.set_title(f'Reference point {Data[0]:.0f} and its {len(nearestArray)} nearest points')
        ax3.legend(prop={'size': 8})
        ax3.set_xlabel('easting')
        ax3.set_ylabel('northing')
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.xaxis.set_tick_params(rotation=30)
        ax3.grid()

        matrixRho = np.vstack([Rho1[nearestArray],Rho2[nearestArray],Rho3[nearestArray],Rho4[nearestArray],Rho5[nearestArray]])
        norm=colors.LogNorm(vmin=10, vmax=1000)
        mat=ax4.matshow(matrixRho, cmap='Spectral_r', norm=norm)
        ax4.axis("equal")
        ax4.set_title('Rhoa')
        ax4.set_xlabel(f'nearest data to the reference point {Data[0]:.0f} ')
        ax4.set_ylabel('spacing')
        ax4.set_ylim([4,0])
        ax4.set_ylim( auto=True)
        #clb = plt.colorbar(mat, orientation='horizontal')
        #clb = ax4.colorbar()
        #clb.ax4.set_title('(ohm.m)',fontsize=10)
        fig.colorbar(mat, ax=ax4, orientation='horizontal')
        
        # plt.xlabel('lines', fontsize=10)
        # plt.ylabel('data', fontsize=10)
        # plt.xticks(ticks=np.arange(len(matrixRho[0])))
        # plt.yticks(ticks=np.arange(len(matrixRho)))
        # # plt.xlim(0,9)
         
        pg.viewer.mpl.showStitchedModels(Inv_indiv.astype(int), ax=ax5, x=None, cMin=10, cMax=1000, cMap='Spectral_r', thk=thk, logScale=True, title='Model (Ohm.m)', zMin=0, zMax=0, zLog=False)

        plt.savefig(pdf, format='pdf')
