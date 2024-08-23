#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:27:04 2024

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

#%% SoilTexture class

class SoilTexture:
    """
    A class to facilitate the analysis and visualization of soil texture and inversion data.
    
    This class uses data processing capabilities to correlate soil properties with inversion results,
    cluster data layers, and generate comprehensive visualizations.
    
    Attributes:
    data_processor : object
        An instance responsible for providing necessary data processing methods and functionalities.
    """

    def __init__(self, data_processor):
        self.data_processor = data_processor

        
    #%% Plot inversion + Rhoa + Soil Type for the Nearest points to Reference points
    def plot_rho_inv_soil_close_to_ref(self,filepath_Processed, filepath_soil, filepath_inv, farmName, farmName_st, kmlFile, output_filename):
        """
        Plots inversion results, resistivity values (Rhoa), and soil types for the nearest points to specified reference points.
        
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # Find nearest points to the reference points
        nearest_points_list, refpoints = self.data_processor.find_nearest_points_to_reference_for_processed_data(filepath_Processed, farmName, kmlFile)
        nearest_points_array = np.array(nearest_points_list)
        Rhoa = nearest_points_array[:, :, 3:-2].astype(float)
    
        # Read soil data from the CSV file
        data_soil = np.genfromtxt(filepath_soil, skip_header=0, delimiter=',', dtype=str, encoding=None)
    
        # Search for 'Field_name' column in the first row
        field_name_index = np.where(data_soil[0] == 'Field_name')[0][0]
    
        # Filter data for the current farmName
        field_data_soil = data_soil[data_soil[:, field_name_index] == farmName_st]
        #field_data_soil_ref = field_data_soil[field_data_soil[:, 7].astype(int) == ref_point]
    
        # Read inversion results from the specified CSV file
        inv_result_data = np.genfromtxt(filepath_inv, delimiter=';', dtype=str)
        # Convert the last column to floats
        inv_result_data[:, -1] = inv_result_data[:, -1].astype(float)
        
        # Get unique values in the last column
        unique_refpoints = np.unique(inv_result_data[:, -1])
        unique_refpoints = np.array(unique_refpoints, dtype=float)
        unique_refpoints.sort()
        unique_refpoints = np.array(unique_refpoints, dtype=str)
    
    
        # Create a dictionary to store data for each unique reference
        inv_result_data_dict = {}
        
        # Iterate over the data and append to the corresponding reference in the dictionary
        for ref in unique_refpoints:
            inv_result_data_dict[ref] = inv_result_data[inv_result_data[:, -1] == ref]
            
        # Get the shape of the first array in inv_result_data_dict to determine the shape of the resulting 3D array
        first_array_shape = inv_result_data_dict[next(iter(inv_result_data_dict))].shape
        # Initialize an empty array to hold the stacked data
        inv_result_data_array = np.empty((len(inv_result_data_dict), *first_array_shape))
        # Iterate over the dictionary and stack the arrays along a new axis
        for idx, (key, value) in enumerate(inv_result_data_dict.items()):
            inv_result_data_array[idx] = value
            

    
        # Create a PDF file for plots
        with PdfPages(output_filename) as pdf:
            # Iterate over each reference point
            for idx, (inv, ref_point, Rhoa) in enumerate(zip(inv_result_data_array, unique_refpoints, nearest_points_array)):
                inv_results_ref_point = inv[:, 3:-1]  # Extract inversion results for the corresponding reference point
                ref_point_str = str(ref_point).split('.')[0]
    
                # Create a plot for the current reference point
                #fig, axes = plt.subplots(1, 3, figsize=(12, 8))
                fig, axes = plt.subplots(1, 3, figsize=(9, 8), gridspec_kw={'width_ratios': [ 0.8, 1, 0.1]})

                fig.suptitle(f'Reference Point {ref_point_str}, Farm Name: {farmName}', fontsize=18)
    
                norm = LogNorm(vmin=10, vmax=1000)
                          

                ########## Plot nearest point data
                if Rhoa is not None:
                    norm = LogNorm(vmin=10, vmax=1000)
                    im_rhoa = axes[0].imshow(Rhoa[:, 3:-2].T, cmap='Spectral_r', norm=norm, interpolation="nearest", aspect='auto')
                    axes[0].set_title(f'Rhoa(processed)')
                    axes[0].set_xlabel('Nearest Soundings')
                    axes[0].set_ylabel('Spacing')
                    x_ticks = np.concatenate((np.arange(-3, 0, 1), np.arange(1, 4, 1)))
                    axes[0].set_xticks(np.arange(len(x_ticks)))
                    axes[0].set_xticklabels(x_ticks)
                    axes[0].set_aspect('auto')
                    axes[0].set_xlim(axes[0].get_xlim())
                    # Add colorbar
                    divider = make_axes_locatable(axes[1])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im_rhoa, cax=cax)
                    
                ########## Plot stitched model of the inversion for all points close to the corresponding reference point
                im_inv = axes[1].imshow(inv_results_ref_point.T, cmap='Spectral_r', norm=norm, interpolation="nearest")
                axes[1].set_title(f'Inversion')
                axes[1].set_xlabel('Nearest Soundings')
                axes[1].set_ylabel('Depth (m)')
                x_ticks = np.concatenate((np.arange(-3, 0, 1), np.arange(1, 4, 1)))
                axes[1].set_xticks(np.arange(len(x_ticks)))
                axes[1].set_xticklabels(x_ticks)
                thk = np.ones(14) * 0.1
                y_ticks = np.arange(0, len(thk) + 1)
                y_tick_labels = np.round(np.arange(0.1, 1.6, 0.1), 1)
                axes[1].set_yticks(y_ticks)
                axes[1].set_yticks(y_ticks + 0.5)  # Add a small offset to position ticks at the bottom
                axes[1].set_yticklabels(y_tick_labels, rotation=0, ha='right', va='bottom')
                axes[1].tick_params(axis='y', which='major')
                
                ########## Plot soil profiles
                if field_data_soil is not None:
                    # Filter data for the current reference point
                    ref_point_str = str(ref_point).split('.')[0]
                    field_data_soil_ref = field_data_soil[field_data_soil[:, 7] == ref_point_str]
                    depths = field_data_soil_ref[:, 12].astype(float)
                    soil_types = field_data_soil_ref[:, 22]

                
                    # Set x-axis height very small
                    axes[2].set_ylim(0, 150)
                    y_ticks = np.arange(0, 1.6, 0.1)
                    axes[2].set_yticks(y_ticks * 100)  # Convert cm to mm for plotting
                    axes[2].set_yticklabels([f'{tick:.1f}' for tick in y_ticks])  # Set the y-tick labels with one decimal degree
                    axes[2].set_ylabel('Depth (m)')  # Set the y-axis label
                    axes[2].set_aspect('auto')  # Reset aspect ratio
                    axes[2].set_yticks([])  # Remove y-axis ticks
                
                    # Remove x-axis label
                    axes[2].set_xlabel(None)
                    axes[2].set_xlabel('')
                    axes[2].set_xlim(0, 0.01)
                    
                    
                # Define color codes for each keyword (HEX_KA5_Group)
                keyword_colors = {
                    'ss': '#fefad9',
                    'ls': '#b18245',
                    'us': '#fbf498',
                    'sl4': '#d6a35f',
                    'sl2': '#fbf498',
                    'sl3': '#fbf498',
                    'll': '#fbf498',
                    'tl': '#954976',
                    'su': '#fbf498',
                    'lu': '#d6a35f',
                    'tu': '#d6a35f',
                    'ut': '#d6a35f',
                    'lt': '#b16b40',
                    'mo': '#73a816',
                    'st': '#fbf498',
                    'tt': '#954976',
                    'ts': '#b18245',
                }
                
                # Create a dictionary to map soil types to colors
                soil_type_colors = {}
                for soil_type in np.unique(data_soil[:, 22]):
                    for keyword, color in keyword_colors.items():
                        if keyword in soil_type.lower():
                            soil_type_colors[soil_type] = color
                            break
                    else:
                        soil_type_colors[soil_type] = 'gray'  # Default color if no keyword match
                
                # Plot soil type column
                if field_data_soil is not None:
                    # Filter data for the current reference point
                    field_data_soil_ref = field_data_soil[field_data_soil[:, 7] == ref_point_str]
                    depths = field_data_soil_ref[:, 12].astype(float)
                    soil_types = field_data_soil_ref[:, 22]
                    # Plot soil types as bars with color gradient
                    for j, (depth, soil_type) in enumerate(zip(depths, soil_types)):
                        color = soil_type_colors[soil_type]  # Get color from soil_type_colors dictionary
                        next_depth = depths[j + 1] if j < len(depths) - 1 else 150  # Next depth or 150 if last depth
                        axes[2].fill_betweenx([depth, next_depth], 0, 1, color=color)
                
                    # Invert y-axis to have surface at the top
                    axes[2].invert_yaxis()
                
                    # Set plot title
                    axes[2].set_title(f'Soil Type')
                
                    # Remove x-axis labels
                    axes[2].set_xlabel('')
                    axes[2].set_xticks([])
                    y_ticks = np.arange(150, -10, -10)
                    axes[2].set_yticks(y_ticks)
    
                    # Add soil type labels inside each bar
                    for depth, soil_type in zip(depths, soil_types):
                        axes[2].text(0.005, depth + 5, soil_type, ha='center', va='top', color='brown')
                
                    # Iterate over soil types to draw dashed lines between them
                    for j in range(1, len(soil_types)):
                        if soil_types[j] != soil_types[j - 1]:
                            axes[2].axhline(depths[j], color='black', linestyle='--', linewidth=1.5)
                
                
                    # Add color bar for inversion result
                    divider_inv = make_axes_locatable(axes[0])
                    cax_inv = divider_inv.append_axes("right", size="5%", pad=0.05)
                    cbar_inv = plt.colorbar(im_inv, cax=cax_inv)
                    # cbar_inv.set_label('Resistivity (ohm.m)')
                
                    # Adjust layout
                    plt.tight_layout()
                
                    # Save the plot to the PDF file
                    pdf.savefig()
                    plt.show()
                
                    # Close the current plot to release memory
                    plt.close()
                
                    print("Plots generated successfully!")
            
    #%% Clustering Algorithms to distinguis different layers
    def cluster_layers(self, filepath_inv, num_components=2):
        """
        Cluster layers in inversion results for different reference points.
    
        Parameters:
        - filepath_inv: str
            Path to the CSV file containing inversion results.
        - num_components: int, optional (default=2)
            Number of components (clusters) for Gaussian Mixture Model.
    
        Returns:
        - cluster_labels_dict: dict
            A dictionary where keys are reference points and values are cluster labels.
        """
        from sklearn.mixture import GaussianMixture
        # Load inversion results from the specified CSV file
        inv_result_data = np.genfromtxt(filepath_inv, delimiter=';', dtype=float)
    
        # Get unique values in the last column
        unique_refpoints = np.unique(inv_result_data[:, -1])
        unique_refpoints = np.array(unique_refpoints, dtype=int)
        unique_refpoints.sort()
    
        # Create a dictionary to store data for each unique reference
        inv_result_data_dict = {}
    
        # Iterate over the data and append to the corresponding reference in the dictionary
        for ref in unique_refpoints:
            inv_result_data_dict[ref] = inv_result_data[inv_result_data[:, -1] == ref]
    
        # Create a dictionary to store cluster labels for each reference point
        cluster_labels_dict = {}
    
        # Iterate over each reference point
        for ref in unique_refpoints:
            # Get the inversion results for the current reference point
            inv_results_ref = inv_result_data_dict[ref]
    
            # Calculate the mean along the columns axis for the current reference point
            mean_inversion_results_ref = np.mean(inv_results_ref[:, 3:-1].astype(float), axis=0)
    
            # Fit Gaussian Mixture Model to the data
            gmm = GaussianMixture(n_components=num_components)
            # Reshape the data to a 2D array
            mean_inversion_results_ref_2d = mean_inversion_results_ref.reshape(-1, 1)
            gmm.fit(mean_inversion_results_ref_2d)
    
            # Get the cluster labels for the current reference point
            cluster_labels_ref = gmm.predict(mean_inversion_results_ref_2d)
    
            # Store the cluster labels for the current reference point in the dictionary
            cluster_labels_dict[ref] = cluster_labels_ref
    
        return cluster_labels_dict
    
    
     #%%
    def correlate_inversion_with_soil_properties(self, filepath_inv, filepath_soil, farmName):
        import numpy as np
        from sklearn.mixture import GaussianMixture
        from scipy.stats import spearmanr
        
        """
        Correlate inversion results with soil data based on MPD, CLAY, SILT, and SAND.
        
        Parameters:
        - filepath_inv: str
            Path to the CSV file containing inversion results.
        - filepath_soil: str
            Path to the CSV file containing soil data.
        - farmName: str
            Farm name.
        
        Returns:
        - cluster_labels_dict: dict
            A dictionary where keys are reference points and values are cluster labels.
        - res_range_forSoilTypes: dict
            A dictionary where keys are soil property values and values are the range of resistivity values.
        """
        
        # Load inversion results from the specified CSV file
        inv_result_data = np.genfromtxt(filepath_inv, delimiter=';', dtype=float)
        
        # Get unique values in the last column
        unique_refpoints = np.unique(inv_result_data[:, -1])
        unique_refpoints = np.array(unique_refpoints, dtype=int)
        unique_refpoints.sort()
        
        # Read soil data from the CSV file
        data_soil = np.genfromtxt(filepath_soil, skip_header=0, delimiter=',', dtype=str, encoding=None)
        
        # Search for 'Field_name' column in the first row
        field_name_index = np.where(data_soil[0] == 'Field_name')[0][0]
        
        # Filter data for the current farmName
        field_data_soil = data_soil[data_soil[:, field_name_index] == farmName]
        
        # Find the indices of the columns of interest
        mpd_index = np.where(data_soil[0] == 'MPD')[0][0]
        clay_index = np.where(data_soil[0] == 'CLAY')[0][0]
        silt_index = np.where(data_soil[0] == 'SILT')[0][0]
        sand_index = np.where(data_soil[0] == 'SAND')[0][0]
        
        # Create dictionaries to store cluster labels and resistivity ranges for each reference point
        cluster_labels_dict = {ref: {} for ref in unique_refpoints}
        res_range_soil_properties = {'MPD': {}, 'CLAY': {}, 'SILT': {}, 'SAND': {}}
        
        combined_matrix_All = []
        all_mpd_values = []
        all_clay_values = []
        all_silt_values = []
        all_sand_values = []
        
        # Open a text file to save the correlation results
        with open(f'soil_corr_MPD-CLAY-SILT-SAND_{farmName}.txt', 'w') as f:
            # Iterate over each reference point
            for ref in unique_refpoints:
                # Filter inversion results for the current reference point
                inv_results_ref = inv_result_data[inv_result_data[:, -1] == ref]
                
                # Calculate the mean along the columns axis for the current reference point
                mean_inversion_results_ref = np.mean(inv_results_ref[:, 3:-1].astype(float), axis=0)
                
                # Filter soil data for the current reference point
                field_data_soil_ref = field_data_soil[field_data_soil[:, 7] == str(ref)]
                
                # Ensure that all soil data exists before proceeding
                if (field_data_soil_ref.size > 0 and 
                    np.all(field_data_soil_ref[:, [mpd_index, clay_index, silt_index, sand_index]] != '')):
                    
                    # Extract soil property values for this reference point
                    mpd_values = field_data_soil_ref[:, mpd_index].astype(float)
                    clay_values = field_data_soil_ref[:, clay_index].astype(float)
                    silt_values = field_data_soil_ref[:, silt_index].astype(float)
                    sand_values = field_data_soil_ref[:, sand_index].astype(float)
                    
                    # Create the combined matrix
                    combined_matrix = np.column_stack((
                        mean_inversion_results_ref,
                        mpd_values,
                        clay_values,
                        silt_values,
                        sand_values
                    ))
                    
                    combined_matrix_All.append(combined_matrix)
                    all_mpd_values.append(mpd_values)
                    all_clay_values.append(clay_values)
                    all_silt_values.append(silt_values)
                    all_sand_values.append(sand_values)
            
            if combined_matrix_All:  # Check if there is any valid data to proceed with
                # Stack all the combined matrices into a single 2D matrix
                combined_matrix_All = np.vstack(combined_matrix_All)
                all_mpd_values = np.concatenate(all_mpd_values)
                all_clay_values = np.concatenate(all_clay_values)
                all_silt_values = np.concatenate(all_silt_values)
                all_sand_values = np.concatenate(all_sand_values)
                
                # Fit Gaussian Mixture Model to the data
                gmm = GaussianMixture(n_components=4)
                gmm.fit(combined_matrix_All)
                
                # Get the cluster labels for all the combined data
                cluster_labels_all = gmm.predict(combined_matrix_All)
                
                start_idx = 0
                
                # Process and analyze inversion results for each unique reference point
                for ref in unique_refpoints:
                    # Filter inversion results for the current reference point
                    inv_results_ref = inv_result_data[inv_result_data[:, -1] == ref]
                    
                    # Calculate the mean along the columns axis for the current reference point
                    mean_inversion_results_ref = np.mean(inv_results_ref[:, 3:-1].astype(float), axis=0)
                    
                    # Filter soil data for the current reference point
                    field_data_soil_ref = field_data_soil[field_data_soil[:, 7] == str(ref)]
                    
                    # Ensure that all soil data exists before proceeding
                    if (field_data_soil_ref.size > 0 and 
                        np.all(field_data_soil_ref[:, [mpd_index, clay_index, silt_index, sand_index]] != '')):
                        
                        # Extract soil property values for this reference point
                        mpd_values = field_data_soil_ref[:, mpd_index].astype(float)
                        clay_values = field_data_soil_ref[:, clay_index].astype(float)
                        silt_values = field_data_soil_ref[:, silt_index].astype(float)
                        sand_values = field_data_soil_ref[:, sand_index].astype(float)
                        
                        # Calculate the number of rows for the current reference point
                        num_rows = len(mean_inversion_results_ref)
                        
                        # Get the cluster labels for the current reference point
                        end_idx = start_idx + num_rows
                        cluster_labels_ref = cluster_labels_all[start_idx:end_idx]
                        start_idx = end_idx
                        
                        # Store the cluster labels for the current reference point in the dictionary
                        cluster_labels_dict[ref] = cluster_labels_ref
                        
                        # Extract resistivity ranges for each soil property value (to be used for the soil texture map)
                        for property_name, property_values in zip(
                            ['MPD', 'CLAY', 'SILT', 'SAND'], 
                            [all_mpd_values, all_clay_values, all_silt_values, all_sand_values]):
                            
                            for prop in np.unique(property_values):
                                if prop not in res_range_soil_properties[property_name]:
                                    res_range_soil_properties[property_name][prop] = [np.inf, -np.inf]
                                
                                # Indexing the relevant part of combined_matrix_All
                                prop_indices = property_values == prop
                                res_of_similar_prop = combined_matrix_All[prop_indices][:, 0]  # Only consider resistivity values
                                
                                res_range_soil_properties[property_name][prop][0] = min(
                                    res_range_soil_properties[property_name][prop][0], 
                                    np.min(res_of_similar_prop))
                                res_range_soil_properties[property_name][prop][1] = max(
                                    res_range_soil_properties[property_name][prop][1], 
                                    np.max(res_of_similar_prop))
                        
                        for property_name, property_values in zip(
                            ['MPD', 'CLAY', 'SILT', 'SAND'], 
                            [mpd_values, clay_values, silt_values, sand_values]):
                            
                            if len(np.unique(property_values)) == 1 and len(np.unique(cluster_labels_ref)) == 1:
                                correlation_coefficient = 1  # Manually set correlation coefficient to 1
                                p_value = 0  # Manually set p-value to 0
                            else:
                                # Calculate Spearman's correlation coefficient between inversion results and property values
                                correlation_coefficient, p_value = spearmanr(cluster_labels_ref, property_values)
    
                            # Print correlation coefficient
                            print(f"Spearman's Correlation Coefficient for {property_name} at Reference Point {ref}: {correlation_coefficient:.2f}, P-value: {p_value:.2f}")
                          
                            # Check if the correlation is statistically significant
                            if p_value < 0.05:
                                print(f"The observed correlation is statistically significant for {property_name} at Reference Point {ref}.")
                            else:
                                print(f"The observed correlation is not statistically significant for {property_name} at Reference Point {ref}.")
                            print()
                            
                            
                            # Write correlation coefficient and p-value to the text file
                            f.write(f"Reference Point {ref} ({property_name}): Correlation Coefficient = {correlation_coefficient:.2f}, P-value = {p_value:.2f}\n")

                        
        return cluster_labels_dict, res_range_soil_properties



     #%% plot_soil_texture_maps based on MPD, CLAY, SILT, and SAND.
    def plot_soil_texture_maps(self, inv_res_filepath, res_range_soil_properties, farmName):
        """
        Generates a multipage PDF with inversion results for each 10 cm depth layer for MPD, CLAY, SILT, and SAND.
        
        Parameters:
        - inv_res_filepath: str
            Path to the CSV file containing inversion results.
        - res_range_soil_properties: dict
            A dictionary where keys are soil property names ('MPD', 'CLAY', 'SILT', 'SAND') and values are dictionaries with 
            resistivity ranges for each property value.
        - pdf_filename: str
            The name of the output PDF file.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Load data directly from the file using np.genfromtxt
        inv_result_data = np.genfromtxt(inv_res_filepath, delimiter=';', skip_header=1)
      
        # Get unique values in the last column
        unique_refpoints = np.unique(inv_result_data[:, -1])
        unique_refpoints = np.array(unique_refpoints, dtype=int)
        unique_refpoints.sort()
        
        pdf_filename = f"soil_texture_maps_MPD_CLAY_SILT_SAND_{farmName}.pdf"

        with PdfPages(pdf_filename) as pdf:
            # Iterate over layers
            Rho_values = inv_result_data[:, 3:-3]
            for layer_index in range(len(Rho_values[0])):
                # Extract true resistivity values for the current layer
                resistivity_values = inv_result_data[:, layer_index + 3]  # Assuming resistivity values start from the fourth column
                
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f"z={layer_index * 0.1:.1f}-{(layer_index + 1) * 0.1:.1f} m _ {farmName}", fontsize=16)
                
                for ax, (property_name, res_range_dict) in zip(axs.ravel(), res_range_soil_properties.items()):
                    # Create a dictionary to map resistivity values to soil property values
                    property_values_for_layer = np.full_like(resistivity_values, np.nan, dtype=float)
                    
                    # Assign soil property values based on resistivity ranges
                    for prop_value, res_range in res_range_dict.items():
                        lower_bound, upper_bound = res_range
                        mask = (resistivity_values >= lower_bound) & (resistivity_values <= upper_bound)
                        property_values_for_layer[mask] = prop_value
                    
                    # Plot the current soil property
                    scatter = ax.scatter(inv_result_data[:, 0], inv_result_data[:, 1], c=property_values_for_layer, cmap='Spectral_r',  s=3)
                    ax.set_title(f"{property_name}")
                    ax.set_xlabel("X Coordinate")
                    ax.set_ylabel("Y Coordinate")
                    ax.axis("equal")
                    fig.colorbar(scatter, ax=ax, label=f'{property_name} Values')
                    ax.grid(True)
                plt.show()

                # Save the current figure to the PDF
                pdf.savefig(fig)
                plt.close(fig)
                
#%%
    
    
    
    
         
      