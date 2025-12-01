# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:40:53 2025

@author: Robotics
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import glob 
from utils import *



metrics_folder = 'Metrics'

metrics = os.listdir('Metrics')

normal_metrics = []

metrics_combined= []

metricas_last = []

metrics_vis_FMC_texture = ['metrics_vis_FMC_texture_20k.csv',
                           'metrics_vis_FMC_texture_80k.csv',
                           'metrics_vis_FMC_texture_195k.csv',]

metrics_vis = ['metrics_vis_20k.csv',
               'metrics_vis_50k.csv']

metrics_texture = ['metrics_texture_20k.csv',
                   'metrics_texture_35k.csv',
                   'metrics_texture_50k.csv']

metrics_vis_texture = ['metrics_vis_texture_20k.csv',
                       'metrics_vis_texture_50k.csv',
                       'metrics_vis_texture_75k.csv',
                       'metrics_vis_texture_100k.csv',
                       'metrics_vis_texture_150k.csv',
                       'metrics_vis_texture_194k.csv']


title = 'Texture features'   
fig_name = 'results_by_model.pdf'
directory = os.path.join('Paper/Figures',fig_name)           
metrics = metrics_texture
metric_name = 'mae'

data_input_metrics = [metrics_vis_FMC_texture,
                      metrics_vis_texture,
                      metrics_vis,
                      metrics_texture]

kinds_of_data = ['lwc_vis_texture',
                'vis_texture',
                'vis',
                'texture']

for metrics, kind_of_data in zip(data_input_metrics, kinds_of_data):
    combined_metrics = []
    for metric in metrics:
        
        metric_path = os.path.join(metrics_folder,metric)
        
        name = metric.split('_')[-1].replace('.csv','')
        
        if os.path.isdir(metric_path):
            continue
        
        metric_df = pd.read_csv(metric_path)
        metric_df = metric_df.iloc[:,1:]
        metric_df = metric_df[['model','mae','rmse','r2']]
        metric_df['features'] = name
       
        
        
        
        metrics_combined.append(metric_df)
        
    combined_metrics = pd.concat(metrics_combined, ignore_index=True)
    combined_metrics['data_input']=kind_of_data

    split_data = combined_metrics['model'].str.rsplit('_',n=1,expand=True)
    combined_metrics['VI']=split_data[0]
    combined_metrics['ML_model']=split_data[1]

    idx_max_r2 = combined_metrics.groupby('model')['r2'].idxmax()
    
    best_per_dataset = combined_metrics.loc[idx_max_r2]
    ridge_indices = best_per_dataset[(best_per_dataset['ML_model']=='Ridge' )|(best_per_dataset['VI']=='MAX')].index
    
    metrics_final = best_per_dataset.drop(ridge_indices)
    
    metrics_final['VI']=metrics_final['VI'].replace('SRWI2','SWRI2')
    
    metricas_last.append(metrics_final)
    
metrics_final = pd.concat(metricas_last,ignore_index=True)
    
    

metrics_MLP = metrics_final[metrics_final['ML_model']=='MLP']
metrics_RF = metrics_final[metrics_final['ML_model']=='RF']
metrics_XGB = metrics_final[metrics_final['ML_model']=='XGB']
metrics_SVR = metrics_final[metrics_final['ML_model']=='SVR']

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

configure_plots()

indices = np.arange(1,25)

figure, ax = plt.subplots(2,2,figsize=(4.4,6))

ax[0,0].plot(metrics_MLP_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[0,0].plot(metrics_MLP_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[0,0].plot(metrics_MLP_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[0,0].plot(metrics_MLP_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[0,0].legend(fontsize = 6)
ax[0,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,0].set_yticks (np.arange(1,25))
ax[0,0].set_yticklabels(metrics_MLP['VI'].to_list())
ax[0,0].tick_params(axis='x', labelsize=6)
ax[0,0].tick_params(axis='y', labelsize=6)
ax[0,0].grid()
ax[0,0].set_xlabel('Correlation factor' )
ax[0,0].set_title('Results for MLP')

ax[0,1].plot(metrics_RF_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[0,1].plot(metrics_RF_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[0,1].plot(metrics_RF_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[0,1].plot(metrics_RF_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[0,1].legend(fontsize = 6)
ax[0,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0,1].set_yticks (np.arange(1,25))
ax[0,1].set_yticklabels(metrics_MLP['VI'].to_list())
ax[0,1].tick_params(axis='x', labelsize=6)
ax[0,1].tick_params(axis='y', labelsize=6)
ax[0,1].grid()
ax[0,1].set_xlabel('Correlation factor' )
ax[0,1].set_title('Results for RF')

ax[1,1].plot(metrics_XGB_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[1,1].plot(metrics_XGB_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[1,1].plot(metrics_XGB_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[1,1].plot(metrics_XGB_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[1,1].legend(fontsize = 6)
ax[1,1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,1].set_yticks (np.arange(1,25))
ax[1,1].set_yticklabels(metrics_MLP['VI'].to_list())
ax[1,1].tick_params(axis='x', labelsize=6)
ax[1,1].tick_params(axis='y', labelsize=6)
ax[1,1].grid()
ax[1,1].set_xlabel('Correlation factor' )
ax[1,1].set_title('Results for XGB')

ax[1,0].plot(metrics_SVR_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='LWC+VIs+Tex')
ax[1,0].plot(metrics_SVR_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs+Tex')
ax[1,0].plot(metrics_SVR_lwc_vi_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='VIs')
ax[1,0].plot(metrics_SVR_tex['r2'], indices, marker = 'o', markersize=2, linestyle='', label='Texture')
ax[1,0].legend(fontsize = 6)
ax[1,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1,0].set_yticks (np.arange(1,25))
ax[1,0].set_yticklabels(metrics_MLP['VI'].to_list())
ax[1,0].tick_params(axis='x', labelsize=6)
ax[1,0].tick_params(axis='y', labelsize=6)
ax[1,0].grid()
ax[1,0].set_xlabel('Correlation factor' )
ax[1,0].set_title('Results for SVR')


figure.tight_layout()
figure.savefig(directory)




#%%


metrics_MLP = metrics_order[metrics_order['ML_model']=='MLP']
metrics_RF = metrics_order[metrics_order['ML_model']=='RF']
metrics_XGB = metrics_order[metrics_order['ML_model']=='XGB']
metrics_SVR = metrics_order[metrics_order['ML_model']=='SVR']



#%%
figure, ax = plt.subplots(1,1,figsize=(2.3,3))
ax.plot(metrics_MLP[metric_name], indices, marker = 'o', markersize=2, linestyle='', label='MLP')
ax.plot(metrics_RF[metric_name], indices, marker = 'o', markersize=2, linestyle='', label='RF')
ax.plot(metrics_XGB[metric_name], indices, marker = 'o', markersize=2, linestyle='', label = 'XGB')
ax.plot(metrics_SVR[metric_name], indices, marker = 'o', markersize=2, linestyle='', label = 'SVR')
ax.legend(fontsize = 6)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks (np.arange(1,25))
ax.set_yticklabels(metrics_MLP['VI'].to_list())
ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=6)
ax.grid()
ax.set_xlabel('Mean average error' )
ax.set_title(title)
figure.tight_layout()
figure.savefig(directory)





#plt.plot(metrics_no_ridge['r2'], marker='o', markersize = 2, linestyle='')

#%%
metrics
    
    
   
    


#%%
