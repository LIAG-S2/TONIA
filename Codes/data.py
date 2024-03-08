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
from sklearn.ensemble import IsolationForest
from matplotlib.backends.backend_pdf import PdfPages interpolate

from Data_Processing import DataProcessing
from Data_Processing import LineDetector
#%%
# Create an instance of DataProcessing
data_processor = DataProcessing()  

# Define file paths and names
filepath_Original = 'TRB_Wertheim_Geophilus_roh_221125.csv'
farmName_Original = 'Trebbin Wertheim_Processed'

# Import reference points from KML file
kmlFile = 'TRB_Wertheim_Referenzpunkte_gemessen_2022-12-01.kml'
refPoints = data_processor.import_reference_points_from_kml(kmlFile, filepath_Original)

# Define spacing labels
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

# Read original data
data_original = data_processor.read_data(filepath_Original)
downsampled_data = data_original[::2]

# Line detection and assign one line to each point
minPoints = 10
maxAngleDeviation = 31
data_line_assigned = data_processor.line_detection_and_assign_lines(downsampled_data, farmName_Original, minPoints, maxAngleDeviation)

# Plot detected lines
data_processor.plot_detected_lines(data_line_assigned, farmName_Original, refPoints)

# Plot detected lines separately
data_processor.plot_lines_separately(data_line_assigned, farmName_Original, refPoints, maxAngleDeviation, minPoints)

# Plot original data's areal plot
data_type_org = 'Original'
data_org = data_processor.plot_data(filepath_Original, spacing_labels, farmName_Original, refPoints1, data_type_org)
data_org = data_processor.plot_data_subplots(filepath_Original, spacing_labels, farmName_Original, refPoints, data_type_org)

# Perform harmfit on original and draw areal plots
data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"
harmfit_onOrg = data_processor.harmfit_fn(data_lineIndexed_file, spacing_labels, farmName_Original, refPoints, data_type_org)

# Perform zscore on original data and draw areal plots
zscore_data = data_processor.zscore_outliers(data_lineIndexed_file, farmName_Original, spacing_labels, refPoints)

# Perform harmfit on zscore result and draw areal plots
zscore_data_file = f" zscore_{farmName_Original}.txt"
data_type_zScore = 'zscore'
harmfit_onZscore = data_processor.harmfit_fn(zscore_data_file, spacing_labels, farmName_Original, refPoints, data_type_zScore)

# LOF on Original
LOF = data_processor.LOF_outliers(data_lineIndexed_file, spacing_labels, refPoints, farmName_Original)

# Harmfit on LOF
LOF_data_file = f"LOF_{farmName_Original}.txt"
data_type_LOF = 'LOF'
harmfit_onLOF = data_processor.harmfit_fn(LOF_data_file, spacing_labels, farmName_Original, refPoints, data_type_LOF)

# Subplot plot HLOF
harmfit_onLOF_file = f"harmfitted_{data_type_LOF}_{farmName_Original}.txt"
data_harmfit_LOF = data_processor.plot_procesed_subplots(harmfit_onLOF_file, spacing_labels, farmName_Original, refPoints, data_type_LOF)


# Define file paths and farm names
# Call the function to compare harmfit on LOF vs original
data_type_LOF = 'LOF'
data_processor.plot_harmfit_vs_original(data_lineIndexed_file, farmName_Original, data_type_LOF)

# Import reference points from KML file
data_processor.reference_points_from_kml(kmlFile, farmName_Original)
