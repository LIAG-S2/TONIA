#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:33:43 2024

@author: sadegh
"""
import csv
import utm
import fiona
import numpy as np
import pandas as pd
import pygimli as pg
import seaborn as sns
import geopandas as gpd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pygimli.frameworks import harmfit
from Data_Processing import LineDetector
from Data_Processing import DataProcessing
from sklearn.ensemble import IsolationForest
from matplotlib.backends.backend_pdf import PdfPages 

#%% inputs
# Define file paths and names
filepath = 'BDW_Seeschlag_Geophilus_roh_2022-10-20.csv'
farmName = 'BDW_Seeschlag'

# Import reference points from KML file
kmlFile = 'BW45701_Referenzpunkte_gemessen.kml'

#%% data processing commands
# Create an instance of DataProcessing
data_processor = DataProcessing()  

# Import reference points from KML file
refPoints = data_processor.import_reference_points_from_kml(kmlFile, farmName)

# Define a hypothetical ref point
# refEutm, refNutm = [385150, 5801200]
# refName = 1
# Stack the arrays horizontally
# refPoints = np.column_stack([refName, refEutm, refNutm])

# Find nearest points to reference
nearest_points_list, refpoints = data_processor.find_nearest_points_to_reference(filepath, farmName, kmlFile)

# Define spacing labels
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

# Read original data
data = data_processor.read_data(filepath)
downsampled_data = data[::2]

# Line detection and assign one line to each point
minPoints = 5
maxAngleDeviation = 31
data_line_assigned = data_processor.line_detection_and_assign_lines(downsampled_data, farmName, minPoints, maxAngleDeviation)

# Plot detected lines
data_processor.plot_detected_lines(data_line_assigned, farmName, refPoints)

# Plot detected lines separately
data_processor.plot_lines_separately(data_line_assigned, farmName, refPoints, maxAngleDeviation, minPoints)

# Plot original data's areal plot
data_type_org = 'Original'
data_org = data_processor.plot_data_subplots(filepath, spacing_labels, farmName, refPoints, data_type_org)

# Perform harmfit on original and draw areal plots
pg.setLogLevel(40)
data_lineIndexed_file = f"data_lineIndexed_{farmName}.txt"
harmfit_onOrg = data_processor.harmfit_fn(data_lineIndexed_file, spacing_labels, farmName, refPoints, data_type_org)

# Perform zscore on original data and draw areal plots
zscore_data = data_processor.zscore_outliers(data_lineIndexed_file, farmName, spacing_labels, refPoints)

# Perform harmfit on zscore result and draw areal plots
zscore_data_file = f" zscore_{farmName}.txt"
data_type_zScore = 'zscore'
harmfit_onZscore = data_processor.harmfit_fn(zscore_data_file, spacing_labels, farmName, refPoints, data_type_zScore)

# LOF on Original
LOF = data_processor.LOF_outliers(data_lineIndexed_file, spacing_labels, refPoints, farmName)

# Harmfit on LOF
LOF_data_file = f"LOF_{farmName}.txt"
data_type_LOF = 'LOF'
harmfit_onLOF = data_processor.harmfit_fn(LOF_data_file, spacing_labels, farmName, refPoints, data_type_LOF)

# Subplot plot HLOF
harmfit_onLOF_file = f"harmfitted_{data_type_LOF}_{farmName}.txt"
data_harmfit_LOF = data_processor.plot_procesed_subplots(harmfit_onLOF_file, spacing_labels, farmName, refPoints, data_type_LOF)

# Define file paths and farm names
# Call the function to compare harmfit on LOF vs original
data_type_LOF = 'LOF'
data_processor.plot_harmfit_vs_original(data_lineIndexed_file, farmName, data_type_LOF)

# Import reference points from KML file
data_processor.import_reference_points_from_kml(kmlFile, farmName)

# plot saunding
selected_indices = [400]
data_processor.plot_sounding_curves(filepath, selected_indices, farmName, kmlFile)
