#! /usr/bin/env python3
# -*- Coding: UTF-8 -*-

r"""
Draw IGM map scatter.
delta_g inter only.
"""



# ####################################################################################
# import joblib
# import warnings
# from sklearn.exceptions import DataConversionWarning

# # 忽略特定类型的警告
# warnings.filterwarnings("ignore", category=DataConversionWarning)
# ###########修改字体
# from matplotlib import font_manager
# # 查找系统中所有可用的Times New Roman字体的路径
# times_new_roman = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# # 从列表中找到一个指定的字体名称
# t_nr_path = [f for f in times_new_roman if 'Times New Roman' in f]

# if t_nr_path:
    # # 如果找到了Times New Roman字体，设置为默认字体
    # prop = font_manager.FontProperties(fname=t_nr_path[0])
    # plt.rcParams['font.family'] = prop.get_name()
# else:
    # # 如果没有找到，使用另一种可用的衬线字体
    # plt.rcParams['font.family'] = 'serif'

# from matplotlib import font_manager, pyplot as plt

# # 查找系统中所有可用的Times New Roman字体的路径
# times_new_roman = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# # 从列表中找到一个指定的字体名称，并确保字体能支持加粗
# t_nr_path = [f for f in times_new_roman if 'Times New Roman' in f and 'Bold' in f]

# if t_nr_path:
    # # 如果找到了Times New Roman Bold字体，设置为默认字体
    # prop = font_manager.FontProperties(fname=t_nr_path[0])
    # plt.rcParams['font.family'] = prop.get_name()
    # plt.rcParams['font.weight'] = 'bold'  # 设置字体为加粗
# else:
    # # 如果没有找到加粗的Times New Roman，尝试设置为普通的Times New Roman并加粗
    # t_nr_path = [f for f in times_new_roman if 'Times New Roman' in f]
    # if t_nr_path:
        # prop = font_manager.FontProperties(fname=t_nr_path[0])
        # plt.rcParams['font.family'] = prop.get_name()
        # plt.rcParams['font.weight'] = 'bold'
    # else:
        # # 如果没有找到Times New Roman，使用默认的衬线字体并设置为加粗
        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['font.weight'] = 'bold'
# #################################################################################



import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
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

# 设置 Times New Roman 字体属性（18号加粗）
font_properties = FontProperties(family='Times New Roman', size=18, weight='bold')

fig, ax = plt.subplots(figsize = (9.6, 7.2))
sc = ax.scatter(sl2r, dg, c = sl2r, s = 1, vmin = -0.05, vmax = 0.05, cmap = cmap_bgr)
ax.set_xlim(-0.05, 0.05)
ax.set_ylim(0, 0.05)
ax.set_xticks(np.linspace(-0.05, 0.05, 11))
ax.set_yticks(np.linspace(0, 0.05, 6))

# 设置 X 和 Y 轴的标签字体
ax.set_xlabel('$\\mathrm{sign}\\left(\\lambda_2\\right)\\rho$ (a.u.)', fontproperties=font_properties)
ax.set_ylabel(u'$\u03b4g_{inter}$ (a.u.)', fontproperties=font_properties)

cbar = plt.colorbar(sc)
cbar.set_ticks(np.linspace(-0.05, 0.05, 11))

# 保存图像
# fig.savefig('IGMmap.ps')
fig.savefig('IGMmap.png', dpi=600)
time_e = time.time()
# plt.show()
print('Time used for drawing:      %5.1lf s' % (time_e - time_m))
print('Total time elapsed:         %5.1lf s' % (time_e - time_s))

