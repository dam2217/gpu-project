# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:12:37 2021

@author: dimitra 
"""

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import pyqtgraph as pg

cpu_result = np.load('./RL_result.npy')
cpu_result = cpu_result[-1]

cpu_sum = np.sum(cpu_result,axis=0)
fig, ax = plt.subplots()
im1 = ax.imshow(cpu_sum)
plt.show()
plt.title('Deconvolution Using CPU',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# gpu_result = np.load('./result.npy')
# gpu_result = gpu_result[-1]

# gpu_sum = np.sum(gpu_result,axis=0)
# im2 = ax.imshow(gpu_sum)
# plt.show()
# plt.title('Deconvolution Using GPU',fontsize=40)

# pg.image(gpu_result)


