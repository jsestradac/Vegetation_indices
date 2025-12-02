# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 00:25:18 2025

@author: sebas
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import glob 
from utils import *





metrics_lwc_vis_texture = pd.read_csv('Metrics/a_lwc_vis_texture.csv')
metrics_vis_texture = pd.read_csv('Metrics/a_vis_texture.csv')
metrics_vis = pd.read_csv('Metrics/a_vis.csv')
metrics_texture = pd.read_csv('Metrics/a_texture.csv')

mm = [metrics_lwc_vis_texture,
      metrics_vis_texture,
      metrics_vis,
      metrics_texture]

metrics_combined = pd.concat(mm, ignore_index=True)

metrics_sorted = metrics_combined.sort_values(['ML_model','VI','r2'])


metrics_MLP = metrics_sorted[metrics_sorted['ML_model']=='MLP']
metrics_RF = metrics_sorted[metrics_sorted['ML_model']=='RF']
metrics_SVR = metrics_sorted[metrics_sorted['ML_model']=='SVR']
metrics_XGB = metrics_sorted[metrics_sorted['ML_model']=='XGB']

metrics_MLP_lwc_vi_tex = metrics_MLP[metrics_MLP['data_input']=='lwc_vis_texture']
metrics_MLP_vi_tex = metrics_MLP[metrics_MLP['data_input']=='vis_texture']
metrics_MLP_vi = metrics_MLP[metrics_MLP['data_input']=='vis']
metrics_MLP_tex = metrics_MLP[metrics_MLP['data_input']=='texture']

metrics_RF_lwc_vi_tex = metrics_RF[metrics_RF['data_input']=='lwc_vis_texture']
metrics_RF_vi_tex = metrics_RF[metrics_RF['data_input']=='vis_texture']
metrics_RF_vi = metrics_RF[metrics_RF['data_input']=='vis']
metrics_RF_tex = metrics_RF[metrics_RF['data_input']=='texture']

metrics_XGB_lwc_vi_tex = metrics_XGB[metrics_XGB['data_input']=='lwc_vis_texture']
metrics_XGB_vi_tex = metrics_XGB[metrics_XGB['data_input']=='vis_texture']
metrics_XGB_vi = metrics_XGB[metrics_XGB['data_input']=='vis']
metrics_XGB_tex = metrics_XGB[metrics_XGB['data_input']=='texture']

metrics_SVR_lwc_vi_tex = metrics_SVR[metrics_SVR['data_input']=='lwc_vis_texture']
metrics_SVR_vi_tex = metrics_SVR[metrics_SVR['data_input']=='vis_texture']
metrics_SVR_vi = metrics_SVR[metrics_SVR['data_input']=='vis']
metrics_SVR_tex = metrics_SVR[metrics_SVR['data_input']=='texture']



indices = np.arange(1,25)

figure, ax = plt.subplots(2,2,figsize=(4.4,6))

ax[0,0].plot(metrics_MLP_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[0,0].plot(metrics_MLP_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[0,0].plot(metrics_MLP_vi['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[0,0].plot(metrics_MLP_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[0,0].legend(fontsize = 6)
ax[0,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,0].set_yticks (np.arange(1,25))
ax[0,0].set_yticklabels(metrics_MLP_vi['VI'].to_list())
ax[0,0].tick_params(axis='x', labelsize=6)
ax[0,0].tick_params(axis='y', labelsize=6)
ax[0,0].grid()
ax[0,0].set_xlabel('Correlation factor' )
ax[0,0].set_title('Results for MLP')

ax[0,1].plot(metrics_RF_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[0,1].plot(metrics_RF_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[0,1].plot(metrics_RF_vi['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[0,1].plot(metrics_RF_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[0,1].legend(fontsize = 6)
ax[0,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,1].set_yticks (np.arange(1,25))
ax[0,1].set_yticklabels(metrics_MLP_vi['VI'].to_list())
ax[0,1].tick_params(axis='x', labelsize=6)
ax[0,1].tick_params(axis='y', labelsize=6)
ax[0,1].grid()
ax[0,1].set_xlabel('Correlation factor' )
ax[0,1].set_title('Results for RF')

ax[1,1].plot(metrics_XGB_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[1,1].plot(metrics_XGB_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[1,1].plot(metrics_XGB_vi['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[1,1].plot(metrics_XGB_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[1,1].legend(fontsize = 6)
ax[1,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,1].set_yticks (np.arange(1,25))
ax[1,1].set_yticklabels(metrics_MLP_vi['VI'].to_list())
ax[1,1].tick_params(axis='x', labelsize=6)
ax[1,1].tick_params(axis='y', labelsize=6)
ax[1,1].grid()
ax[1,1].set_xlabel('Correlation factor' )
ax[1,1].set_title('Results for XGB')

ax[1,0].plot(metrics_SVR_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[1,0].plot(metrics_SVR_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[1,0].plot(metrics_SVR_vi['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[1,0].plot(metrics_SVR_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[1,0].legend(fontsize = 6)
ax[1,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,0].set_yticks (np.arange(1,25))
ax[1,0].set_yticklabels(metrics_MLP_vi['VI'].to_list())
ax[1,0].tick_params(axis='x', labelsize=6)
ax[1,0].tick_params(axis='y', labelsize=6)
ax[1,0].grid()
ax[1,0].set_xlabel('Correlation factor' )
ax[1,0].set_title('Results for SVR')


figure.tight_layout()
