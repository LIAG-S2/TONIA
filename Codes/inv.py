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
from inversion import VESRhoModelling, VES2dModelling
from inversion import InversionClass


#%%  Create an instance of the inversion class
inversion_obj = InversionClass()

#%% 1D Inversion for all lines

# data_file = 'TRB_Wertheim_Geophilus_roh_221125.csv'
# farmName = 'Trebbin Wertheim'
farmName_Original = 'Trebbin Wertheim_Processed'
# farmName_Processed = 'Boossen_1601'
# farmName_Processed = 'Boossen_1211'
# farmName_Original = 'Trebbin Aussenschlag'
# farmName_Original = 'Großbeeren_1'
# farmName_Original = 'Beerfelde Fettke_Original'

data_lineIndexed_file = f"data_lineIndexed_{farmName_Original}.txt"

# inv txt file name
# Inv_File_name = 'TRB_Aussenschlag_geophilus_eri_epsg4236'
Inv_File_name = 'TRB_Wertheim_geophilus_eri_epsg4236'
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

filepath_Original = 'TRB_Wertheim_Geophilus_roh_221125.csv'
kmlFile = 'TRB_Wertheim_Referenzpunkte_gemessen_2022-12-01.kml'
refPoints = inversion_obj.import_reference_points_from_kml(kmlFile, filepath_Original)

#  inv for original data
amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
# Stmodels, chi2 = inversion_1D_all_lines(data_lineIndexed_file, amVec, farmName_Original)

#  inv for processed (HLOF)
farmName_Processed = 'Trebbin Wertheim_HLOF'
# farmName_Original = 'Großbeeren_1'
data_type_HLOF = 'LOF'
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName_Original}.txt"
Stmodels, chi2 = inversion_obj.inversion_1D_all_lines(filepath_Processed, amVec, farmName_Processed, Inv_File_name)

#%% 1D inversion plotting for all lines 

# Example usage:
amVec = np.arange(1, 6) * 0.6  # Geophilius 3.0 (2020)
thk = np.ones(15) * 0.1

farmName = 'Trebbin Wertheim_Processed'
data_lineIndexed_file = f"data_lineIndexed_{farmName}.txt"


filepath_Original = 'TRB_Wertheim_Geophilus_roh_221125.csv'
kmlFile = 'TRB_Wertheim_Referenzpunkte_gemessen_2022-12-01.kml'
refPoints = inversion_obj.import_reference_points_from_kml(kmlFile, filepath_Original)
 
# inv plot for processed (HLOF)
farmName_Processed = 'Trebbin Wertheim_HLOF'
data_type_HLOF = 'LOF'
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName}.txt"
inversion_obj.plot_1d_inversion_results_allLines(filepath_Processed, amVec, farmName_Processed, data_type_HLOF, thk)

#%% inversion result subplot
farmName_Processed = 'Trebbin Wertheim_HLOF'
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
inversion_obj.subplot_inversion_results(inv_res_filepath, farmName_Processed)

#%% inversion result subplot chi2<20
farmName_Processed = 'Trebbin Wertheim_HLOF'
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
Inv_File_name = 'TRB_Wertheim_geophilus_eri_epsg4236'
inversion_obj.subplot_inversion_limit_chi2(inv_res_filepath, farmName_Processed, Inv_File_name, chi2_limit=20)


#%% inversion result multipage pdf
farmName_Processed = 'Trebbin Wertheim_HLOF'
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
inversion_obj.plot_inv_results_multipage_pdf(inv_res_filepath, farmName_Processed)

#%% plot Chi2
farmName_Processed = 'Trebbin Wertheim_HLOF'
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
inversion_obj.plot_chi2_results(inv_res_filepath, farmName_Processed)
