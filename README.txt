# Installation Guide

To get started with this project, follow the steps below to set up your environment. There are multiple methods to set up the required Python environment, including using Anaconda or installing packages individually with pip. This guide covers both methods.

## Step 1: Installing Python

Before starting, make sure you have Python installed on your machine. You can install Python in several ways:

1. **Using Anaconda (Recommended):**
   - Anaconda is a distribution of Python, R, and other tools that simplifies package management and deployment. It's ideal for scientific computing and data science projects.
   - To install Anaconda, follow the instructions here: [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/).

2. **Using Spyder (Standalone IDE):**
   - Spyder is an open-source Python IDE that comes with Anaconda but can also be installed separately.
   - To install Spyder, follow the instructions here: [Spyder Installation Guide](https://docs.spyder-ide.org/current/installation.html).

## Step 2: Installing Required Libraries

1. **Installing pyGIMLi:**
   - Follow the installation instructions on the pyGIMLi website using the following link: [pyGIMLi Installation Guide](https://www.pygimli.org/installation.html).

2. **Install the required libraries using pip:**
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

## Step 3: Verifying the Installation

After installing all the required packages, it's a good idea to verify that everything is set up correctly. You can do this by running a simple Python script to import the libraries:

```python
import utm
import fiona
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
