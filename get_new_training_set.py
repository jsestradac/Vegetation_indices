# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 18:15:48 2025

@author: Robotics
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

pytorch_data = pd.read_csv('Data/fmc_d_annotated_images.csv')
pytorch_data_order = pytorch_data.sort_values(['leaf_type','leaf_number','set','drying_stage','spectral_number'])
my_data = pd.read_csv('Data/test_total.csv')
my_data_simple = my_data[['Species','Stages','FMC_d', 'blue','green','red','nir','red_edge']]
my_data_simple_order = my_data_simple.sort_values(['Species','Stages'])
#%%
import os
def get_leaf_tag(x):
    blue_name = os.path.basename(x)
    species = x.split('/')[0]
    leaf_tag = species+'_'+blue_name.split('_')[0]
    return leaf_tag

my_data_simple_order['tag']= my_data_simple_order['blue'].apply(get_leaf_tag)
#%%
pytorch_data_order['tag'] = (
    pytorch_data_order['leaf_type'] + '_' +
    pytorch_data_order['image_file'].str.split('_').str[0]
)
tags_list = my_data_simple_order['tag']
training_mask = pytorch_data_order['tag'].isin(tags_list) 

pytorch_data_order.loc[training_mask,'set']='test'
pytorch_data_order.loc[~training_mask,'set']='training'
