#!/usr/bin/env python

import glob
import os, sys
import colormap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}

ndvi_image = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\ccfc_images\2022_06_16\Photo1_2022_06_16_13_00_35.jpg'
im = Image.open(ndvi_image)
imArray = np.array(im).astype(np.uint8)

# Take the last row of pixels, which shows the scale
r = imArray[-1, :, 0]
g = imArray[-1, :, 1]
b = imArray[-1, :, 2]
ndvi = np.linspace(-1.0, 1.0, r.size)

ndvi_1 = ndvi[ndvi < 0]
r_1 = np.linspace(56, 12, ndvi_1.size)
g_1 = np.linspace(18, 208, ndvi_1.size)
b_1 = np.linspace(118, 235, ndvi_1.size)

ndvi_2 = ndvi[(ndvi >= 0) & (ndvi < 0.38)]
r_2 = np.linspace(12, 233, ndvi_2.size)
g_2 = np.linspace(154, 233, ndvi_2.size)
b_2 = np.linspace(12, 12, ndvi_2.size)

ndvi_3 = ndvi[(ndvi >= 0.38) & (ndvi < 0.75)]
r_3 = np.linspace(233, 236, ndvi_3.size)
g_3 = np.linspace(233, 14, ndvi_3.size)
b_3 = np.linspace(12, 12, ndvi_3.size)

ndvi_4 = ndvi[ndvi >= 0.75]
r_4 = np.linspace(236, 236, ndvi_4.size)
g_4 = np.linspace(14, 14, ndvi_4.size)
b_4 = np.linspace(12, 223, ndvi_4.size)

ndvi_step = np.hstack([ndvi_1, ndvi_2, ndvi_3, ndvi_4])
r_step = np.hstack([r_1, r_2, r_3, r_4])
g_step = np.hstack([g_1, g_2, g_3, g_4])
b_step = np.hstack([b_1, b_2, b_3, b_4])

fig = plt.figure()
fig.set_size_inches((6, 4))
ax = plt.axes([0.15, 0.15, 0.80, 0.80])
ax.set_ylabel('RGB value')
ax.set_xlabel('NDVI')
ax.plot(ndvi, r, color='r', alpha=0.2, linewidth=1)
ax.plot(ndvi, g, color='g', alpha=0.2, linewidth=1)
ax.plot(ndvi, b, color='b', alpha=0.2, linewidth=1)
ax.plot(ndvi_step, r_step, color='r', linewidth=1, ls='--')
ax.plot(ndvi_step, g_step, color='g', linewidth=1, ls='--')
ax.plot(ndvi_step, b_step, color='b', linewidth=1, ls='--')
plt.savefig(r'ndvi_rgb_scale.png', dpi=300)
