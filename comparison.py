# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:57:15 2021

@author: dimit
"""
import numpy as np
import pyqtgraph as pg

gpu_result = np.load('./result.npy')
# print('gpu:',gpu_result)
# pg.image(gpu_result[-1,...])

max_val = np.amax(gpu_result)

cpu_result = np.load('./RL_result.npy')
max_cpu = np.amax(cpu_result)

bcpu = np.load('./backward_cpu.npy')
max_bcpu = np.amax(bcpu)

fcpu = np.load('./forward_cpu.npy')
max_fcpu = np.amax(fcpu)

bgpu = np.load('./backward_result.npy')
max_bgpu = np.amax(bgpu)

fgpu = np.load('./forward_result.npy')
max_fgpu = np.amax(fgpu)
