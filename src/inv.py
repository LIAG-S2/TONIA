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
from inversion import InversionClass
from matplotlib.colors import LogNorm
from pygimli.frameworks import harmfit
from Data_Processing import DataProcessing
from sklearn.ensemble import IsolationForest
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from inversion import VESRhoModelling, VES2dModelling

#%% inputs
# Define file paths and names
farmName = 'BDW_Seeschlag'
farmName_Processed = 'BDW_Seeschlag_Processed'
filepath_Original = 'BDW_Seeschlag_Geophilus_roh_2022-10-20.csv'
Inv_File_name = 'BDW_ Seeschlag_geophilus_eri_epsg4236'
# Import reference points from KML file
kmlFile = 'BW45701_Referenzpunkte_gemessen.kml'

#%% First, create an instance of DataProcessing
# Create an instance of the inversion class 

data_processor = DataProcessing()  
inversion_obj = InversionClass(data_processor)

#%% 1D Inversion for all lines


# inv txt file name
# Inv_File_name = 'TRB_Aussenschlag_geophilus_eri_epsg4236'
# Inv_File_name = 'TRB_Wertheim_geophilus_eri_epsg4236'
# Inv_File_name = 'BFD_ Fettke_geophilus_eri_epsg4236'
# Inv_File_name = 'BFD_ Banane_geophilus_eri_epsg4236'
# Inv_File_name = 'LPP_1211_geophilus_eri_epsg4236'
# Inv_File_name = 'LPP_1601_geophilus_eri_epsg4236'
# Inv_File_name = 'BDW_ Seeschlag_geophilus_eri_epsg4236'
# Inv_File_name = 'BDW_Schröder_geophilus_eri_epsg4236'
# Inv_File_name = 'IGZ_trasse_geophilus_eri_epsg4326'

# # Define a hypothetical ref point
# refEutm, refNutm = [385489, 5801147]
# refName = 1
# # Stack the arrays horizontally
# refPoints = np.column_stack([refName, refEutm, refNutm])
# # Define header and stack header with ref_Points
# header = ['Name', 'E', 'N']
# refPointsTable = np.vstack((header, refPoints))



refPoints = inversion_obj.import_reference_points_from_kml(kmlFile, farmName)
# Define a hypothetical ref point
# refEutm, refNutm = [385489, 5801147]
# refName = 1
# Stack the arrays horizontally
# refPoints = np.column_stack([refName, refEutm, refNutm])


# Find Survey Depth
import os
# Extract the filename without extension
filename = os.path.splitext(os.path.basename(filepath_Original))[0]

# Split the filename by underscores and hyphens
filename_parts = filename.split('_')

# Look for parts that match the date format
for part in filename_parts:
    if len(part) == 10:  # Check if the part has the length of a date in the format yyyy-mm-dd
        try:
            # Try to parse the part as a date
            survey_date = part
            break
        except ValueError:
            pass

# If no date is found, set survey_date to None
else:
    survey_date = None

#  inv for processed (HLOF)

data_type_HLOF = 'LOF'
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName}.txt"
Stmodels, chi2 = inversion_obj.inversion_1D_all_lines(filepath_Processed, farmName_Processed, Inv_File_name, survey_date)

 #%% 1D inversion plotting for all lines 
# Example usage:
thk1 = np.ones(19) * 0.1

# inv plot for processed (HLOF)
data_type_HLOF = 'LOF'
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName}.txt"
inversion_obj.plot_1d_inversion_results_allLines(filepath_Processed, farmName_Processed, data_type_HLOF, thk1, kmlFile)

#%% inversion result subplot
lam =30
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
inversion_obj.subplot_inversion_results(inv_res_filepath, farmName_Processed)

#%% inversion result subplot chi2<20
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
inversion_obj.subplot_inversion_limit_chi2(inv_res_filepath, farmName_Processed, Inv_File_name, chi2_limit=20)


#%% inversion result multipage pdf
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
inversion_obj.plot_inv_results_multipage_pdf(inv_res_filepath, farmName_Processed)

#%% plot Chi2
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
inversion_obj.plot_chi2_results(inv_res_filepath, farmName_Processed)
# %% compare ERi1 and rhoa1
# data info
# Create an instance of DataProcessing
data_processor = DataProcessing()  
data_type_org = 'Original'
spacing_labels = ['Spacing 1', 'Spacing 2', 'Spacing 3', 'Spacing 4', 'Spacing 5']

refPoints = data_processor.import_reference_points_from_kml(kmlFile, filepath_Original)
data_type_LOF = 'LOF'
harmfit_onLOF_filepath = f"harmfitted_{data_type_LOF}_{farmName}.txt"

# inv resultt
lam =30
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'

inversion_obj.subplot_compare_data_and_inversion_result(farmName, filepath_Original, harmfit_onLOF_filepath, inv_res_filepath)

#%% plot_3d_invResult
farmName_Processed = farmName_Processed
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
# Set the view angle
elev=25
azim=250
inversion_obj.plot_3d_invResult(inv_res_filepath, farmName_Processed, elev, azim)

#%% plot_3d_invResult with desired cutoff
easting_cutoff= 300
northing_cutoff= 300
elev=25
azim=250
inversion_obj.plot_3d_invResult_section_maker(filepath=inv_res_filepath, 
                                                  farm_name=farmName_Processed, 
                                                  elev=elev, 
                                                  azim=azim, 
                                                  easting_cutoff=easting_cutoff, 
                                                  northing_cutoff=northing_cutoff)
#%% Plot inversion for nearest point to ref points
harmfit_onLOF_file = f"harmfitted_{data_type_LOF}_{farmName}.txt"
inversion_obj.plot_inv_column_close_to_refs(harmfit_onLOF_file, farmName, kmlFile)
