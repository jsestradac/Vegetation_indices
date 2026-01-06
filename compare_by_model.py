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




#configure_plots()
metrics_lwc_vis_texture = pd.read_csv('Metrics/a_lwc_vis_texture.csv')
metrics_vis_texture = pd.read_csv('Metrics/a_vis_texture.csv')
metrics_vis = pd.read_csv('Metrics/a_fmc_vis.csv')
metrics_texture = pd.read_csv('Metrics/a_texture.csv')

#metrics_vis.loc[metrics_vis['VI']]

mm = [metrics_lwc_vis_texture,
      metrics_vis_texture,
      metrics_vis,
      metrics_texture]

metrics_combined = pd.concat(mm, ignore_index=True)

metrics_sorted = metrics_combined.sort_values(['r2','VI','ML_model'])


metrics_MLP = metrics_sorted[metrics_sorted['ML_model']=='MLP']
metrics_RF = metrics_sorted[metrics_sorted['ML_model']=='RF']
metrics_SVR = metrics_sorted[metrics_sorted['ML_model']=='SVR']
metrics_XGB = metrics_sorted[metrics_sorted['ML_model']=='XGB']




metrics_XGB_lwc_vi_tex = metrics_XGB[metrics_XGB['data_input']=='lwc_vis_texture']
metrics_XGB_lwc_vi_tex_order = metrics_XGB_lwc_vi_tex.sort_values(by='r2')
order_index = metrics_XGB_lwc_vi_tex_order['VI'].to_list()

metrics_XGB_vi_tex = metrics_XGB[metrics_XGB['data_input']=='vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_XGB_vi = metrics_XGB[metrics_XGB['data_input']=='vis'].set_index('VI').reindex(order_index).reset_index()
metrics_XGB_tex = metrics_XGB[metrics_XGB['data_input']=='texture'].set_index('VI').reindex(order_index).reset_index()


metrics_MLP_lwc_vi_tex = metrics_MLP[metrics_MLP['data_input']=='lwc_vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_MLP_vi_tex = metrics_MLP[metrics_MLP['data_input']=='vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_MLP_vi = metrics_MLP[metrics_MLP['data_input']=='vis'].set_index('VI').reindex(order_index).reset_index()
metrics_MLP_tex = metrics_MLP[metrics_MLP['data_input']=='texture'].set_index('VI').reindex(order_index).reset_index()

metrics_RF_lwc_vi_tex = metrics_RF[metrics_RF['data_input']=='lwc_vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_RF_vi_tex = metrics_RF[metrics_RF['data_input']=='vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_RF_vi = metrics_RF[metrics_RF['data_input']=='vis'].set_index('VI').reindex(order_index).reset_index()
metrics_RF_tex = metrics_RF[metrics_RF['data_input']=='texture'].set_index('VI').reindex(order_index).reset_index()



metrics_SVR_lwc_vi_tex = metrics_SVR[metrics_SVR['data_input']=='lwc_vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_SVR_vi_tex = metrics_SVR[metrics_SVR['data_input']=='vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_SVR_vi = metrics_SVR[metrics_SVR['data_input']=='vis'].set_index('VI').reindex(order_index).reset_index()
metrics_SVR_tex = metrics_SVR[metrics_SVR['data_input']=='texture'].set_index('VI').reindex(order_index).reset_index()

y_labels = metrics_XGB_lwc_vi_tex_order['VI'].to_list()
index1 = y_labels.index('SWRI')
index2 = y_labels.index('SWRI1')
index3 = y_labels.index('SWRI2')
index4 = y_labels.index('NDWI')

y_labels[index1] = 'SRWI'
y_labels[index2] = 'SRWI1'
y_labels[index3] = 'SRWI2'
y_labels[index4] = 'NDII'

indices = np.arange(1,25)

figure, ax = plt.subplots(2,2,figsize=(7.27,6.5))

ax[0,0].plot(metrics_MLP_lwc_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[0,0].plot(metrics_MLP_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[0,0].plot(metrics_MLP_vi['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[0,0].plot(metrics_MLP_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[0,0].legend(fontsize = 7, loc = 0)
ax[0,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,0].set_yticks (np.arange(1,25))
ax[0,0].set_yticklabels(y_labels)
ax[0,0].tick_params(axis='x', labelsize=7)
ax[0,0].tick_params(axis='y', labelsize=7)
ax[0,0].grid()
ax[0,0].set_xlabel(f'Coefficient of determination\na)',fontsize = 9 )
ax[0,0].set_title('Results for MLP',fontsize = 9)

ax[0,1].plot(metrics_RF_lwc_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[0,1].plot(metrics_RF_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[0,1].plot(metrics_RF_vi['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[0,1].plot(metrics_RF_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[0,1].legend(fontsize = 7, loc = 0)
ax[0,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,1].set_yticks (np.arange(1,25))
ax[0,1].set_yticklabels(y_labels)
ax[0,1].tick_params(axis='x', labelsize=7)
ax[0,1].tick_params(axis='y', labelsize=7)
ax[0,1].grid()
ax[0,1].set_xlabel(f'Coefficient of determination\nb)',fontsize = 9 )
ax[0,1].set_title('Results for RFR',fontsize = 9)

ax[1,1].plot(metrics_XGB_lwc_vi_tex_order['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[1,1].plot(metrics_XGB_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[1,1].plot(metrics_XGB_vi['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[1,1].plot(metrics_XGB_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[1,1].legend(fontsize = 7, loc = 0 )
ax[1,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,1].set_yticks (np.arange(1,25))
ax[1,1].set_yticklabels(y_labels)
ax[1,1].tick_params(axis='x', labelsize = 7)
ax[1,1].tick_params(axis='y', labelsize = 7)
ax[1,1].grid()
ax[1,1].set_xlabel(f'Coefficient of determination\nd)',fontsize = 9 )
ax[1,1].set_title('Results for XGB',fontsize = 9)

ax[1,0].plot(metrics_SVR_lwc_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[1,0].plot(metrics_SVR_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[1,0].plot(metrics_SVR_vi['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[1,0].plot(metrics_SVR_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[1,0].legend(fontsize = 7, loc = 0)
ax[1,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,0].set_yticks (np.arange(1,25))
ax[1,0].set_yticklabels(y_labels)
ax[1,0].tick_params(axis='x', labelsize = 7)
ax[1,0].tick_params(axis='y', labelsize = 7)
ax[1,0].grid()
ax[1,0].set_xlabel(f'Coefficient of determination\nc)',fontsize = 9)
ax[1,0].set_title('Results for SVR',fontsize = 9)


figure.tight_layout()

figure.savefig('Paper/Figures/Comparison_by_model.pdf')

#%%
indices = np.arange(1,25)



metrics_MLP = metrics_sorted[metrics_sorted['ML_model']=='MLP']
metrics_RF = metrics_sorted[metrics_sorted['ML_model']=='RF']
metrics_SVR = metrics_sorted[metrics_sorted['ML_model']=='SVR']
metrics_XGB = metrics_sorted[metrics_sorted['ML_model']=='XGB']




metrics_XGB_lwc_vi_tex = metrics_XGB[metrics_XGB['data_input']=='lwc_vis_texture']
metrics_XGB_lwc_vi_tex_order = metrics_XGB_lwc_vi_tex.sort_values(by='r2')
order_index = metrics_XGB_lwc_vi_tex_order['VI'].to_list()
metrics_MLP_lwc_vi_tex = metrics_MLP[metrics_MLP['data_input']=='lwc_vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_RF_lwc_vi_tex = metrics_RF[metrics_RF['data_input']=='lwc_vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_SVR_lwc_vi_tex = metrics_SVR[metrics_SVR['data_input']=='lwc_vis_texture'].set_index('VI').reindex(order_index).reset_index()

y_labels1 = metrics_XGB_lwc_vi_tex_order['VI'].to_list()
index1 = y_labels1.index('SWRI')
index2 = y_labels1.index('SWRI1')
index3 = y_labels1.index('SWRI2')
index4 = y_labels1.index('NDWI')

y_labels1[index1] = 'SRWI'
y_labels1[index2] = 'SRWI1'
y_labels1[index3] = 'SRWI2'
y_labels1[index4] = 'NDII'

metrics_SVR_vi_tex = metrics_SVR[metrics_SVR['data_input']=='vis_texture']
metrics_SVR_vi_tex_order = metrics_SVR_vi_tex.sort_values(by='r2')
order_index = metrics_SVR_vi_tex_order['VI'].to_list()
metrics_XGB_vi_tex = metrics_XGB[metrics_XGB['data_input']=='vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_MLP_vi_tex = metrics_MLP[metrics_MLP['data_input']=='vis_texture'].set_index('VI').reindex(order_index).reset_index()
metrics_RF_vi_tex = metrics_RF[metrics_RF['data_input']=='vis_texture'].set_index('VI').reindex(order_index).reset_index()

y_labels2 = metrics_SVR_vi_tex_order['VI'].to_list()
index1 = y_labels2.index('SWRI')
index2 = y_labels2.index('SWRI1')
index3 = y_labels2.index('SWRI2')
index4 = y_labels2.index('NDWI')

y_labels2[index1] = 'SRWI'
y_labels2[index2] = 'SRWI1'
y_labels2[index3] = 'SRWI2'
y_labels2[index4] = 'NDII'


metrics_SVR_vi = metrics_SVR[metrics_SVR['data_input']=='vis']
metrics_SVR_vi_order = metrics_SVR_vi.sort_values(by='r2')
order_index = metrics_SVR_vi_order['VI'].to_list()
metrics_XGB_vi = metrics_XGB[metrics_XGB['data_input']=='vis'].set_index('VI').reindex(order_index).reset_index()
metrics_RF_vi = metrics_RF[metrics_RF['data_input']=='vis'].set_index('VI').reindex(order_index).reset_index()
metrics_MLP_vi = metrics_MLP[metrics_MLP['data_input']=='vis'].set_index('VI').reindex(order_index).reset_index()

y_labels3 = metrics_SVR_vi_order['VI'].to_list()
index1 = y_labels3.index('SWRI')
index2 = y_labels3.index('SWRI1')
index3 = y_labels3.index('SWRI2')
index4 = y_labels3.index('NDWI')

y_labels3[index1] = 'SRWI'
y_labels3[index2] = 'SRWI1'
y_labels3[index3] = 'SRWI2'
y_labels3[index4] = 'NDII'


metrics_SVR_tex = metrics_SVR[metrics_SVR['data_input']=='texture']
metrics_SVR_tex_order = metrics_SVR_tex.sort_values(by='r2')
order_index = metrics_SVR_tex_order['VI'].to_list()
metrics_MLP_tex = metrics_MLP[metrics_MLP['data_input']=='texture'].set_index('VI').reindex(order_index).reset_index()
metrics_XGB_tex = metrics_XGB[metrics_XGB['data_input']=='texture'].set_index('VI').reindex(order_index).reset_index()
metrics_RF_tex = metrics_RF[metrics_RF['data_input']=='texture'].set_index('VI').reindex(order_index).reset_index()

y_labels4 = metrics_SVR_tex_order['VI'].to_list()
index1 = y_labels4.index('SWRI')
index2 = y_labels4.index('SWRI1')
index3 = y_labels4.index('SWRI2')
index4 = y_labels4.index('NDWI')

y_labels4[index1] = 'SRWI'
y_labels4[index2] = 'SRWI1'
y_labels4[index3] = 'SRWI2'
y_labels4[index4] = 'NDII'


figure, ax = plt.subplots(2,2,figsize=(7.27,7))

ax[0,0].plot(metrics_MLP_lwc_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='MLP')
ax[0,0].plot(metrics_RF_lwc_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='RFR')
ax[0,0].plot(metrics_SVR_lwc_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='SVR')
ax[0,0].plot(metrics_XGB_lwc_vi_tex_order['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='XGB')
ax[0,0].legend(fontsize = 7, loc = 0)
ax[0,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,0].set_yticks (np.arange(1,25))
ax[0,0].set_yticklabels(y_labels1)
ax[0,0].tick_params(axis='x', labelsize = 7)
ax[0,0].tick_params(axis='y', labelsize = 7)
ax[0,0].grid()
ax[0,0].set_xlabel(f'Coefficient of determination\na)',fontsize = 9)
ax[0,0].set_title('LWC + VIs + Tex features',fontsize = 9)

ax[0,1].plot(metrics_MLP_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='MLP')
ax[0,1].plot(metrics_RF_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='RFR')
ax[0,1].plot(metrics_SVR_vi_tex_order['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='SVR')
ax[0,1].plot(metrics_XGB_vi_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='XGB')
ax[0,1].legend(fontsize = 7, loc = 0)
ax[0,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,1].set_yticks (np.arange(1,25))
ax[0,1].set_yticklabels(y_labels2)
ax[0,1].tick_params(axis='x', labelsize = 7)
ax[0,1].tick_params(axis='y', labelsize = 7)
ax[0,1].grid()
ax[0,1].set_xlabel(f'Coefficient of determination\nb',fontsize = 9 )
ax[0,1].set_title('VIs + Tex Features',fontsize = 9)

ax[1,1].plot(metrics_MLP_vi['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='MLP')
ax[1,1].plot(metrics_RF_vi['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='RFR')
ax[1,1].plot(metrics_SVR_vi_order['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='SVR')
ax[1,1].plot(metrics_XGB_vi['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='XGB')
ax[1,1].legend(fontsize = 7, loc = 0 )
ax[1,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,1].set_yticks (np.arange(1,25))
ax[1,1].set_yticklabels(y_labels3)
ax[1,1].tick_params(axis='x', labelsize = 7)
ax[1,1].tick_params(axis='y', labelsize = 7)
ax[1,1].grid()
ax[1,1].set_xlabel(f'Coefficient of determination\nd)',fontsize = 9 )
ax[1,1].set_title('VIs Features',fontsize = 9)

ax[1,0].plot(metrics_MLP_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='MLP')
ax[1,0].plot(metrics_RF_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='RFR')
ax[1,0].plot(metrics_SVR_tex_order['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='SVR')
ax[1,0].plot(metrics_XGB_tex['r2'].clip(lower=0), indices, marker = 'o', markersize=2, linestyle='', label='XGB')
ax[1,0].legend(fontsize = 7, loc = 2)
ax[1,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,0].set_yticks (np.arange(1,25))
ax[1,0].set_yticklabels(y_labels4)
ax[1,0].tick_params(axis='x', labelsize = 7)
ax[1,0].tick_params(axis='y', labelsize = 7)
ax[1,0].grid()
ax[1,0].set_xlabel(f'Coefficient of determination\nc)',fontsize = 9 )
ax[1,0].set_title('Texture Features',fontsize = 9)


figure.tight_layout()

figure.savefig('Paper/Figures/Comparison_by_features.pdf')
