# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 17:09:13 2025

@author: Robotics
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data_mobile = pd.read_csv('Metrics/mobilenet_lwc.csv')
data_resnet = pd.read_csv('Metrics/resnet_lwc.csv')
data_real = pd.read_csv('Metrics/real_lwc.csv')

#%%

import os

def get_vi(x):
    basename = os.path.basename(x)
    index = basename.split('_')[0]
    return index

def get_model(x):
    basename = os.path.basename(x)
    model = basename.split('_')[1]
    model = model.split('.')[0]
    return model
    
data_real['VI'] = data_real['model'].apply(get_vi)
data_real['ML'] = data_real['model'].apply(get_model)
data_real = data_real[['VI','ML','r2', 'mae','mae_std']]

data_mobile['VI'] = data_mobile['model'].apply(get_vi)
data_mobile['ML'] = data_mobile['model'].apply(get_model)
data_mobile = data_mobile[['VI','ML','r2', 'mae','mae_std']]

data_resnet['VI'] = data_resnet['model'].apply(get_vi)
data_resnet['ML'] = data_resnet['model'].apply(get_model)
data_resnet = data_resnet[['VI','ML','r2', 'mae','mae_std']]

data_real = data_real[data_real['VI'] != 'MAX']
data_resnet = data_resnet[data_resnet['VI'] != 'MAX']
data_mobile = data_mobile[data_mobile['VI'] != 'MAX']
#%%

real_order = data_real.sort_values(by='r2')
order_idx = real_order.index



resnet_order = data_resnet.loc[order_idx]
mobile_order = data_mobile.loc[order_idx]

y_labels = resnet_order['VI'].to_list()

idx1 = y_labels.index('SWRI1')
idx2 = y_labels.index('SWRI')
#idx3 = y_labels.index('SWRI2')
idx4 = y_labels.index('NDWI')

y_labels[idx1] = 'SRWI1'
y_labels[idx2] = 'SRWI'
#y_labels[idx3] = 'SRWI2'
y_labels[idx4] = 'NDII'

indices = np.arange(1,25)

fig, ax = plt.subplots(1,1,figsize = (4.45, 3))
ax.plot(real_order['r2'], indices, marker = 'o', markersize=4, linestyle='', label='Real LWC')
ax.plot(resnet_order['r2'], indices, marker = 'o', markersize=4, linestyle='', label='ResNet50 LWC')
ax.plot(mobile_order['r2'], indices, marker = 'o', markersize=4, linestyle='', label='MobileNetV2 LWC', color = 'r')

ax.set_xticks([0.25, 0.5, 0.75, 1])
ax.set_yticks (np.arange(1,25))
ax.set_yticklabels(y_labels)
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', labelsize=7)
ax.grid()
ax.set_title('Results using as input real and predicted LWC' )
ax.set_xlabel('Coefficient of determination')
ax.legend()
fig.tight_layout()
fig.savefig('Paper/Figures/metrics_models_cnn.pdf')
