#! /usr/bin/env python3
# -*- Coding: UTF-8 -*-

r"""
Draw IGM map scatter.
delta_g inter only.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sys import argv

time_s = time.time()
cmap_bgr = LinearSegmentedColormap.from_list('bgr', ['#0000FF', '#00FF00', '#FF0000'])
dg, sl2r = np.loadtxt(argv[1] if len(argv) > 1 else 'output.txt', unpack = True, usecols = (0, 3))
# column 3 is sl2r, which should not be changed. column 0, 1, 2 are delta_g_inter, delta_g_intra 
# and delta_g, respectively. The default is to use the delta_g_inter.
# here if, by default, use delta_g_inter, the variable 'dg' stands for delta_g_inter.
# if you change the line you used, you'd better change the variable name and y label as well.
time_m = time.time()
print('Time used for reading data: %5.1lf s' % (time_m - time_s))
fig, ax = plt.subplots(figsize = (9.6, 7.2))
sc = ax.scatter(sl2r, dg, c = sl2r, s = 1, vmin = -0.05, vmax = 0.05, cmap = cmap_bgr)
ax.set_xlim(-0.05, 0.05)
ax.set_ylim(0, 0.05)
ax.set_xticks(np.linspace(-0.05, 0.05, 11))
ax.set_yticks(np.linspace(0, 0.05, 6))
ax.set_xlabel('$\\mathrm{sign}\\left(\\lambda_2\\right)\\rho$ (a.u.)')
ax.set_ylabel(u'$\u03b4g_{inter}$ (a.u.)')
cbar = plt.colorbar(sc)
cbar.set_ticks(np.linspace(-0.05, 0.05, 11))
# fig.savefig('IGMmap.ps')
fig.savefig('IGMmap.png')
time_e = time.time()
# plt.show()
print('Time used for drawing:      %5.1lf s' % (time_e - time_m))
print('Total time elapsed:         %5.1lf s' % (time_e - time_s))

