#!/usr/bin/env python
"""

"""
import os
import sys
import glob
import datetime
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': "Arial"}
plt.rcParams.update(params)


# Read in CSV file
# name,area,climate,rain_season,fire_season,pv_season,npv_season


# Plot rain_season vs fire_season as heatmap with numbers
# - Separate humid, semi-arid, and arid dunefields



