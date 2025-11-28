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

metrics_folder = 'Metrics'

metrics = os.listdir('Metrics')

normal_metrics = []

metrics_combined= []

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
               
metrics = metrics_vis_FMC_texture

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
#%%
split_data = combined_metrics['model'].str.rsplit('_',n=1,expand=True)
combined_metrics['Index']=split_data[0]
combined_metrics['Model']=split_data[1]

idx_max_r2 = combined_metrics.groupby('model')['r2'].idxmax()

best_per_dataset = combined_metrics.loc[idx_max_r2]


    
    
   
    


#%%
