# TONIA: Sub-Area Differentiated-Optimized Nutrient Management in Arable Farming

## Project Overview

**TONIA** aims to optimize nutrient management in arable farming to achieve environmental benefits and improve crop yields. Traditional farming methods often apply nutrients uniformly across fields, which can result in suboptimal fertilization and environmental contamination. To address this, the TONIA project uses geoelectric measurements to create detailed 3D soil texture maps, helping to tailor nutrient application to specific soil conditions.

## Project Description

Farms exhibit significant variability in soil composition and moisture, leading to uneven nutrient distribution and suboptimal crop yields. Excessive or insufficient fertilization can negatively impact both yield and the environment. The **TONIA project** addresses this by developing soil sensors and 3D soil texture maps that extend to a depth of 150 cm, focusing on soil moisture. This data, combined with yield modeling, generates detailed nutrient maps with a 10x10m resolution. These maps allow for targeted nutrient application, optimizing fertilizer use according to specific soil conditions.

## Responsibility of LIAG

The **LIAG** (Leibniz Institute for Applied Geophysics) is responsible for evaluating the geoelectric data collected using the **Geophilus Electricus** measuring device. This device measures soil conductivity at various depths, and LIAG's task is to reconstruct these conductivity values into detailed soil texture maps. These maps, along with data from other sensors like gamma activity, provide the foundation for precision nutrient management. The project involves collaboration among farmers, agronomists, soil scientists, geophysicists, and software developers to address practical, socially relevant issues.

## Installation Guide

To set up your environment for this project, follow the instructions below. You can choose between using **Anaconda** for an integrated environment or installing packages individually with `pip`.

### Step 1: Installing Python

Before starting, ensure that Python is installed on your machine. You have two main options:

1. **Using Anaconda (Recommended):**
   - Anaconda is a comprehensive distribution of Python and other tools, ideal for scientific and data science applications.
   - Install Anaconda by following the instructions here: [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/).

2. **Using Spyder (Standalone IDE):**
   - Spyder is an open-source Python IDE that can be installed independently or as part of Anaconda.
   - Install Spyder by following the instructions here: [Spyder Installation Guide](https://docs.spyder-ide.org/current/installation.html).

### Step 2: Installing Required Libraries

1. **Installing pyGIMLi:**
   - Follow the specific installation instructions for pyGIMLi on their website: [pyGIMLi Installation Guide](https://www.pygimli.org/installation.html).

2. **Install the Required Libraries using `pip`:**
   - Open your terminal (Command Prompt/PowerShell on Windows or Terminal on macOS/Linux) and run the following commands:

     ```bash
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
     ```

### Step 3: Verifying the Installation

After installing all the required packages, it is a good idea to verify that everything is set up correctly. You can do this by running a simple Python script to import the libraries:

```python
import utm
import fiona
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from scipy import interpolate
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.ensemble import IsolationForest

print("All libraries are installed and imported successfully!")
