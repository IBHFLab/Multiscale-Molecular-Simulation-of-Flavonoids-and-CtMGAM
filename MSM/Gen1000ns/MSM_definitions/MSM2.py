"""
This module is part of DuIvyProcedures.procedures, designed for dealing MSM. 
Written by 杜艾维.
"""

import os
import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda
from matplotlib import pyplot as plt
from MDAnalysis.analysis.dihedrals import Dihedral
from DuIvyTools.DuIvyTools.FileParser.xvgParser import XVG

from deeptime.decomposition import TICA, vamp_score
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.tools.analysis import mfpt
from deeptime.plots import Network
from deeptime.plots import plot_density
from deeptime.plots import plot_implied_timescales
from deeptime.plots.chapman_kolmogorov import plot_ck_test
from deeptime.util.validation import implied_timescales


import time
import logging
from typing import List, Tuple
from colorama import Back, Fore, Style



####################################################################################
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=DataConversionWarning)
###########修改字体
from matplotlib import font_manager
# 查找系统中所有可用的Times New Roman字体的路径
times_new_roman = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# 从列表中找到一个指定的字体名称
t_nr_path = [f for f in times_new_roman if 'Times New Roman' in f]

if t_nr_path:
    # 如果找到了Times New Roman字体，设置为默认字体
    prop = font_manager.FontProperties(fname=t_nr_path[0])
    plt.rcParams['font.family'] = prop.get_name()
else:
    # 如果没有找到，使用另一种可用的衬线字体
    plt.rcParams['font.family'] = 'serif'

from matplotlib import font_manager, pyplot as plt

# 查找系统中所有可用的Times New Roman字体的路径
times_new_roman = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# 从列表中找到一个指定的字体名称，并确保字体能支持加粗
t_nr_path = [f for f in times_new_roman if 'Times New Roman' in f and 'Bold' in f]

if t_nr_path:
    # 如果找到了Times New Roman Bold字体，设置为默认字体
    prop = font_manager.FontProperties(fname=t_nr_path[0])
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.weight'] = 'bold'  # 设置字体为加粗
else:
    # 如果没有找到加粗的Times New Roman，尝试设置为普通的Times New Roman并加粗
    t_nr_path = [f for f in times_new_roman if 'Times New Roman' in f]
    if t_nr_path:
        prop = font_manager.FontProperties(fname=t_nr_path[0])
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.weight'] = 'bold'
    else:
        # 如果没有找到Times New Roman，使用默认的衬线字体并设置为加粗
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.weight'] = 'bold'
#################################################################################




class log(object):
    """log class, a logging system parent class, provied five functions for output debug, info, warning, error, and critical messages."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    dip_log: str = ""
    program_log: str = ""

    def dump_log(self, outname: str):
        outname = self.check_output_exist(outname)
        with open(outname, "a") as fo:
            fo.write(self.dip_log + "\n")
            fo.write("\n>>>>>> run terminal command log <<<<<<\n")
            fo.write(self.program_log + "\n")

    def debug(self, msg, dip: bool = True):
        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"[Debug] {time_info}\n{msg}"
        if dip:
            self.logger.debug(Fore.CYAN + Back.WHITE + message + Style.RESET_ALL)
            self.dip_log += message + "\n"
        else:
            self.program_log += message + "\n"

    def info(self, msg, dip: bool = True):
        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"[Info] {time_info}\n{msg}"
        if dip:
            self.logger.info(Fore.GREEN + message + Style.RESET_ALL)
            self.dip_log += message + "\n"
        else:
            self.program_log += message + "\n"

    def warn(self, msg, dip: bool = True):
        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"[Warning] {time_info}\n{msg}"
        if dip:
            self.logger.warning(Fore.YELLOW + message + Style.RESET_ALL)
            self.dip_log += message + "\n"
        else:
            self.program_log += message + "\n"

    def error(self, msg, dip: bool = True):
        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"[ERROR] {time_info}\n{msg}"
        self.logger.error(Fore.WHITE + Back.RED + message + Style.RESET_ALL)
        if dip:
            self.dip_log += message + "\n"
        else:
            self.program_log += message + "\n"

    def critical(self, msg, dip: bool = True):
        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"[CRITICAL] {time_info}\n{msg}"
        self.logger.critical(Fore.WHITE + Back.RED + message + Style.RESET_ALL)
        if dip:
            self.dip_log += message + "\n"
        else:
            self.program_log += message + "\n"
        sys.exit()


class MSM(log):
    def __init__(self) -> None:
        self.conf = {}
        self.conf["fig"] = "png"

    def __call__(self) -> None:

        data2start = 10 ## start from 10 rows
        dimension = 2
        lag4tica = 25
        number_clusters = 100
        transition_count_mode = "effective"  ## "effective" #"sliding","sample","effective","sliding-effective"
        lag4ITS_range = [1, 10, 20, 30, 40, 50, 70, 100, 130, 160, 200]
        lag4MSM = 100
        meta_number = 4

        #### TO READ
        ## http://www.emma-project.org/latest/tutorials/notebooks/00-pentapeptide-showcase.html
        ## https://ambermd.org/tutorials/advanced/tutorial41/index.php#3.%20tICA%20Analysis%20and%20MSM%20Construction
        ## http://www.emma-project.org/latest/index.html
        ## https://deeptime-ml.github.io/trunk/index.html

        ## Prepare the data 
        xvg = XVG("dist_rmsd.xvg")
        data_byframe = np.array(xvg.data_columns)[1:, data2start:].T
        time_list = np.array(xvg.data_columns[0][data2start:])
        dt = time_list[1] - time_list[0]

        self.info(f">> dt == {dt} ps")
        self.info(f">> data shape == {data_byframe.shape}")

        ## do TICA
        ## if dimension < 2:
        ##     self.critical(f"dimension should be greater than 1, but got {dimension}")
        ## self.info(f">> dimension == {dimension}, lag4TICA == {lag4tica}")
        ## estimator = TICA(dim=dimension, lagtime=lag4tica).fit(data_byframe)
        ## model = estimator.fetch_model()
        ## data = model.transform(data_byframe)
        ## self.info(f">> data shape after TICA == {data.shape}")
        ## self.info("-"*80 + f"\n>> cumulative variance of TICA : \n{model.cumulative_kinetic_variance[:dimension]}\n" + "-"*80)
        ## vamp = vamp_score(model, r="VAMP2")
        ## self.info("-"*80 + f"\n>> vamp2 score == {vamp} \n" + "-"*80)
        data = data_byframe
        print(data)

        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], alpha=0.2, linewidths=0)
        plt.xlabel("tICA 1", fontsize=20, fontweight='bold')
        plt.ylabel("tICA 2", fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"""tica12.{self.conf["fig"]}""", dpi=600)
        plt.close()
        if dimension > 2:
            plt.figure()
            plt.scatter(data[:, 0], data[:, 2], alpha=0.2, linewidths=0)
            plt.xlabel("tICA 1", fontsize=20, fontweight='bold')
            plt.ylabel("tICA 3", fontsize=20, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"""tica13.{self.conf["fig"]}""", dpi=600)
            plt.close()
            plt.figure()
            plt.scatter(data[:, 1], data[:, 2], alpha=0.2, linewidths=0)
            plt.xlabel("tICA 2", fontsize=20, fontweight='bold')
            plt.ylabel("tICA 3", fontsize=20, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"""tica23.{self.conf["fig"]}""", dpi=600)
            plt.show()
            plt.close()

        plt.figure()
        plt.plot(time_list, data[:, 0], label="tICA 1")
        plt.plot(time_list, data[:, 1], label="tICA 2")
        if dimension > 2:
            plt.plot(time_list, data[:, 2], label="tICA 3")
        plt.ylabel("IC", fontsize=20, fontweight='bold')
        plt.xlabel("Time (ps)", fontsize=20, fontweight='bold')
        plt.legend( loc='upper left') #左上角
        plt.tight_layout()
        plt.savefig(f"""tica.{self.conf["fig"]}""", dpi=600)
        plt.show()
        plt.close()

        ## do k-means
        kmeans = KMeans(n_clusters=number_clusters)
        clustering = kmeans.fit(data).fetch_model()
        assignments = clustering.transform(data)
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=assignments, alpha=0.2, linewidths=0)
        plt.xlabel("tICA 1", fontsize=20, fontweight='bold')
        plt.ylabel("tICA 2", fontsize=20, fontweight='bold')
        #plt.colorbar(label="cluster index")
        cbar = plt.colorbar()
        cbar.set_label("cluster index", fontsize=20, fontweight='bold')  # Set font size and bold for colorbar label
        cc_x = clustering.cluster_centers[:, 0]
        cc_y = clustering.cluster_centers[:, 1]
        plt.scatter(cc_x, cc_y, linewidths=0, marker="o", s=5, c="k", label="cluster centers")
        plt.legend( loc='upper left') #左上角
        plt.tight_layout()
        plt.savefig(f"""clustering.{self.conf["fig"]}""", dpi=600)
        plt.show()
        plt.close()

        ## do ITS
        models = []
        for lagtime in lag4ITS_range:
            counts = TransitionCountEstimator(lagtime=lagtime, count_mode=transition_count_mode).fit_fetch(assignments)
            models.append(MaximumLikelihoodMSM().fit_fetch(counts))
        its_data = implied_timescales(models)
        plt.figure()
        ax = plt.gca()
        plot_implied_timescales(its_data, ax=ax)
        ax.set_yscale('log')
        ax.set_title('Implied timescales', fontsize=20, fontweight='bold')
        ax.set_xlabel('lag time (steps)', fontsize=20, fontweight='bold')
        ax.set_ylabel('timescale (steps)', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"""ITS.{self.conf["fig"]}""", dpi=600)
        plt.show()
        plt.close()

        ## do MSM and ck_test
        counts = TransitionCountEstimator(lagtime=lag4MSM, count_mode=transition_count_mode).fit_fetch(assignments)
        msm = MaximumLikelihoodMSM().fit_fetch(counts)
        self.info(f">> MSM with lagtime={lag4MSM} has {msm.n_states} states")
        # self.info(msm.timescales())
        plt.plot(msm.timescales(), linewidth=0, marker="o")
        plt.xlabel("Index", fontsize=18, fontweight='bold')
        plt.ylabel("Timescale (steps)", fontsize=18, fontweight='bold')
        plt.title(f"Timescales of MSM with lagtime={lag4MSM}", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"""timescales.{self.conf["fig"]}""", dpi=600)
        plt.show()
        plt.close()

        ck_test = msm.ck_test(models, n_metastable_sets=meta_number, err_est=True)
        plt.figure()#
        grid = plot_ck_test(ck_test)
        
        # Iterate over all axes in the figure and modify their fonts
        for ax in plt.gcf().get_axes():
            ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='bold')  # Set X-axis label font size and bold
            ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='bold')  # Set Y-axis label font size and bold
        plt.tight_layout()#
        plt.savefig(f"""ck_test.{self.conf["fig"]}""", dpi=600)
        plt.show()
        plt.close()

        self.info("-"*80 + f"\n>> MSM.state_fraction == {msm.state_fraction}\n" + "-"*80)
        self.info("-"*80 + f"\n>> MSM.count_fraction == {msm.count_fraction}\n" + "-"*80)

        ## do PCCA
        m = msm.pcca(meta_number)
        self.info(f">> PCCA number of metastable states == {m.n_metastable}")
        # print(m.metastable_distributions.shape)
        self.info(f">> PCCA coarse_grained_transition_matrix :\n{m.coarse_grained_transition_matrix}")
        self.info(f">> PCCA coarse_grained_stationary_probability :\n{m.coarse_grained_stationary_probability}")

        for n in range(meta_number):
            plt.plot(m.metastable_distributions[n], label=f"meta{n}", alpha=0.5)
        plt.legend( loc='upper right') #右上角
        plt.xlabel("Cluster index", fontsize=20, fontweight='bold')
        plt.ylabel("Probability", fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"""metastable_distributions.{self.conf["fig"]}""", dpi=600)
        plt.show()
        plt.close()

        ## do MFPT
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from deeptime.plots import Network  # 从 deeptime.plots 导入 Network 类

# class MyClass:
    # def __init__(self, conf):
        # self.conf = conf

    # def info(self, msg):
        # print(msg)

    # def critical(self, msg):
        # print(f"Critical: {msg}")

    # def do_mfpt_analysis(self, m, meta_number, assignments, time_list, data, dt, number_clusters, dimension):
        ## do MFPT
        mfpt_matrix = []
        for n in range(meta_number):
            mfpt_matrix.append(mfpt(m.coarse_grained_transition_matrix, n, tau=1, mu=m.coarse_grained_stationary_probability))
        
        ## convert mfpt data from step to ns 
        ## Mean first passage times from set A to set B, in units of the input trajectory time step.      
        mfpt_matrix = np.array(mfpt_matrix)
        mfpt_matrix = mfpt_matrix *dt /1000   # ns
        self.info(f">> MFPT (mean first passage time) matrix in unit ns: \n{mfpt_matrix}")
        
        inverse_mfpt = np.zeros_like(mfpt_matrix)
        nz = mfpt_matrix.nonzero()
        inverse_mfpt[nz] = 1.0 / mfpt_matrix[nz]
        self.info(f">> Inverse MFPT: \n{inverse_mfpt}")

        self.info(f">> PCCA assignments: \n{m.assignments}")
        if len(m.assignments) != number_clusters:
            self.critical(f"PCCA assignments length {len(m.assignments)} != number_clusters {number_clusters}; normally this means some clusters cannot be assigned to PCCA states, due to insufficient data or poor parameters")
       
        meta_assignments = np.array([m.assignments[a] for a in assignments])
        frame = [i for i in range(len(meta_assignments))]
        df = pd.DataFrame({"frame": frame, "time(ps)": time_list, "cluster_id": assignments, "meta_state": meta_assignments})
        
        for d in range(dimension):
            df[f"tICA{d+1}"] = data[:, d]
       
        for n in range(meta_number):
            dist = [m.metastable_distributions[n][a] for a in assignments]
            df[f"meta_distribution{n+1}"] = dist
        
        df.to_csv("meta_assignments.csv", index=False)
        ## TODO output meta structure

        ## draw network
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
       
       # 绘制散点图（每个meta状态的点）
        for n in range(meta_number):
            meta = meta_assignments == n
            ax.scatter(data[meta, 0], data[meta, 1], alpha=0.1, label=f"meta{n}", linewidths=0)
        
        plt.xlabel("tICA 1", fontdict={"size": 25, "weight": "bold"})
        plt.ylabel("tICA 2", fontdict={"size": 25, "weight": "bold"})
        # 修改XY轴数值字体大小
        plt.tick_params(axis='both', which='major', labelsize=18)  # 使X和Y轴的数值字体变大
        
        # 修改图例字体大小,并将其位置设置为左上角
        plt.legend(fontsize=18)
        
        # 计算每个meta状态的中心和大小
        centers = []
        meta_size = []
        for n in range(meta_number):
            meta = meta_assignments == n
            centers.append(np.mean([data[meta, 0], data[meta, 1]], axis=1))
            meta_size.append(np.sum(meta))
        
        centers = np.array(centers)
        meta_size = np.array(meta_size)
       
        self.info(f">> meta state centers: {centers}")
        self.info(f">> meta state sizes: {meta_size}")
        
        # 创建 Network 对象并绘制网络
        # 在这里，我们使用 deeptime.plots.Network 来绘制网络图
        nw = Network(
           inverse_mfpt, # MFPT矩阵
           pos=centers, # 节点位置
           edge_labels=mfpt_matrix, # 边的标签显示MFPT值
           edge_scale= 2,
           edge_label_format='{:.4f} ns', # 设置MFPT标签的显示格式
           state_sizes=meta_size, # 节点的大小
           edge_curvature=6.0, # 边的弯曲度
           state_scale=1, # 节点的缩放比例
           state_labels=[f"s{n} {meta_size[n]/np.sum(meta_size):.2%}" for n in range(meta_number)]) # 节点标签
        
        # 绘制网络并将MFPT标签放置在每条边上
        nw.plot(ax=ax, size=12, alpha=1)
        
        # # 计算边的中点位置，以调整标签显示
        # for i, (start, end) in enumerate(zip(centers[:-1], centers[1:])):
           # # 计算边的中点
           # midpoint = (start + end) / 2
           # mfpt_value = mfpt_matrix[i, i+1]  # 根据需要选择矩阵中的元素
           # # 设置标签的显示位置
           # ax.text(midpoint[0], midpoint[1], f'{mfpt_matrix[i]:.4f} ns', fontsize=12, color='black', ha='center', va='center')
        
        # 调整布局以避免重叠
        plt.tight_layout()
        
        # 保存图像为文件
        plt.savefig(f"""Network.{self.conf["fig"]}""", dpi=600)
        # 显示图形
        plt.show()
        # 关闭图形
        plt.close()


if __name__ == "__main__":
    MSM()()