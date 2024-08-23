TONIA, sub-area differentiated-optimized nutrient management in arable farming

The aim of this joint project is not only to optimize nutrient management on arable land, but also for environmental reasons. Since soil is laterally variable and has different specifications, the entire farm areas are measured with a geoelectric measuring system called Geophilus Electricus. The responsibility of LIAG is to evaluate the geoelectric data in this project. The conductivity maps are then converted into 3D soil texture maps with the help of reference points' measurements.

Project description
Farms are not uniform in soil composition and moisture. Consequently, plants in a large area do not receive ideal fertilization. With a few amount of nutrients, plants will not achieve the possible yield on one hand. On the other hand, too many fertilizers contaminate the environment. The goal is a precise calculation of the demand for sub-surface areas. The TONIA project partners are developing soil sensors and 3D soil texture maps that explore the soil down to a depth of 150 cm, especially with regard to soil moisture. Combined with yield modelling, nutrient target value maps and maps of the usable field capacity are created with a resolution of 10x10m, on which the amount of fertilizer is converted according to sub-surface areas. Instead of distributing the total amount of nutrients evenly over the field, in future they will be applied as needed.

Responsibility of LIAG
The Geophilus Electricus measuring device (Lück & Rühlmann, 2013) measures continuously with five different penetration depths, by increasing the distances between the electrodes. The sensitivity distribution (Guillemoteau et al., 2017) shows how the measured values are composed of the background resistivities. The task of the LIAG is to reconstruct the conductivities from the measurement data. A robust software, tailored to the routine field application, is to be developed, which generates soil texture maps together with other sensors such as the gamma activity. These then serve as the basis for further management. 
The TONIA project brings together farmers, agronomists, soil scientists, geophysicists and software developers to tackle socially relevant problems in a practical way.


Installation Guide
To get started with this project, follow the steps below to set up your environment. There are multiple methods to set up the required Python environment, including using Anaconda or installing packages individually with pip. This guide covers both methods.
Step 1: Installing Python
Before starting, make sure you have Python installed on your machine. You can install Python in several ways:
1.	Using Anaconda (Recommended):
o	Anaconda is a distribution of Python, R, and other tools that simplifies package management and deployment. It's ideal for scientific computing and data science projects.
o	To install Anaconda, follow the instructions here: Anaconda Installation Guide.
2.	Using Spyder (Standalone IDE):
o	Spyder is an open-source Python IDE that comes with Anaconda but can also be installed separately.
o	To install Spyder, follow the instructions here: Spyder Installation Guide.
Step 2: Installing Required Libraries
1.	Installing pyGIMLi:
o	Follow the installation instructions on the pyGIMLi website using the following link: pyGIMLi Installation Guide.
2.	Install the required libraries using pip:
o	Open your terminal (Command Prompt/PowerShell on Windows or Terminal on macOS/Linux) and run the following commands:
pip install utm
pip install Fiona
pip install scipy
pip install numpy
pip install pandas
pip install seaborn
pip install geopandas
pip install matplotlib
pip install scikit-learn
pip install DateTime


Step 3: Verifying the Installation
After installing all the required packages, it's a good idea to verify that everything is set up correctly. You can do this by running a simple Python script to import the libraries:
import utm
import Fiona
import numpy as np
import pandas as pd
import pygimli as pg
import seaborn as sns
import geopandas as gpd
from scipy import interpolate
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.ensemble import IsolationForest

print("All libraries are installed and imported successfully!")

If this script runs without any errors, your environment is set up correctly.

