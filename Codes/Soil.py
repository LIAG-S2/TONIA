#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:40:46 2024

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
from Soil_Texture import SoilTexture
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
# Import reference points from KML file
kmlFile = 'BW45701_Referenzpunkte_gemessen.kml'
# soil data file
filepath_soil = 'TONIA_soil_data_10cm.csv'
farmName_st = 'Seeschlag' # must be mentioned from column 'Field_name' of the 'filepath_soil'

#%% First, create an instance of DataProcessing
# Create an instance of the inversion class 
data_processor = DataProcessing()  
inversion_obj = InversionClass(data_processor)
soil_texture = SoilTexture(data_processor)

#%% Plot inversion + Rhoa + Soil Type for the Nearest points to Reference points
data_type_HLOF = 'LOF'
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName}.txt"
filepath_inv = f'1Dinv_response_Nearest_{farmName}.csv'
output_filename = f'soil_inversion_plots_{farmName}.pdf'
soil_texture.plot_rho_inv_soil_close_to_ref(filepath_Processed, filepath_soil, filepath_inv, farmName, farmName_st, kmlFile, output_filename)

#%% Correlate inversion results with soil data based on MPD, CLAY, SILT, and SAND.
filepath_Processed = f"harmfitted_{data_type_HLOF}_{farmName}.txt"
filepath_soil = 'TONIA_soil_data_10cm.csv'
filepath_inv = f'1Dinv_response_Nearest_{farmName}.csv'

cluster_labels, res_range_soil_properties = soil_texture.correlate_inversion_with_soil_properties(
    filepath_inv=filepath_inv,
    filepath_soil=filepath_soil,
    farmName=farmName_st
)

#%% plot_soil_texture_maps based on MPD, CLAY, SILT, and SAND.
inv_res_filepath = f'1Dinv_response_all_lines_{farmName_Processed}.csv'
soil_texture.plot_soil_texture_maps(
    inv_res_filepath=inv_res_filepath,
    res_range_soil_properties=res_range_soil_properties,
    farmName=farmName
)
#%%
