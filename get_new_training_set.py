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
my_data_simple = my_data
my_data_simple_order = my_data_simple.sort_values(['Species','Stages'])

print(pytorch_data_order['set'].unique())
#%%
import os
def get_leaf_tag(x):
    blue_name = os.path.basename(x)
    species = x.split('/')[0]
    leaf_tag = species+'_'+blue_name.split('_')[0]
    return leaf_tag

def get_leaf_number(x):
    blue_name = x
    species = x.split('_')[1]
    leaf_tag = int(species.split('d')[0][-3:])
    return leaf_tag

def get_d_stage(x):
    blue_name = x
    species = x.split('_')[1]
    leaf_tag = int(species.split('d')[1])
    return leaf_tag
    

my_data_simple_order['tag']= my_data_simple_order['blue'].apply(get_leaf_tag)
my_data_simple_order['leaf_number']= my_data_simple_order['tag'].apply(get_leaf_number)
my_data_simple_order['stage_number']= my_data_simple_order['tag'].apply(get_d_stage)

my_data_simple_order2 = my_data_simple_order.sort_values(['Species','leaf_number','stage_number'])

my_data_simple_order2.to_csv('Data/total_test_ordered.csv', index = False)
#%%
pytorch_data_order['tag'] = (
    pytorch_data_order['leaf_type'] + '_' +
    pytorch_data_order['image_file'].str.split('_').str[0]
)
tags_list = my_data_simple_order['tag']
training_mask = pytorch_data_order['tag'].isin(tags_list) 

pytorch_data_order.loc[training_mask,'set']='test'
pytorch_data_order.loc[~training_mask, 'set']='train'
pytorch_data_order = pytorch_data_order.drop('tag', axis = 1)

pytorch_data_order.to_csv('Data/fmc_d_annotated_images1.csv')

#%%
import cv2 as cv

img = cv.imread('C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/test/Avocado_leaf004d0_1.tif',-1)
plt.imshow(img)

#%%

import shutil
train_path = 'C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/training'
test_path = 'C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/test'
val_path = 'C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/validation'

test2_path = 'C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/test2'

train_images = os.listdir('C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/training')
test_images = os.listdir('C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/test')
val_images = os.listdir('C:/Users/sebas/Desktop/fmc_prediction/code/water_content_prediction/data/images/validation')


test_files = pytorch_data_order.loc[pytorch_data_order['set']=='test']
test_files['files']= test_files['leaf_type']+'_'+test_files['image_file']

files = test_files['files'].to_list()

a=0

for file in train_images:
    if file in files:
        old_file = os.path.join(train_path,file)
        new_file = os.path.join(test2_path,file)
        shutil.move(old_file,new_file)
        a = a+1
        
for file in test_images:
    if file in files:
        old_file = os.path.join(test_path,file)
        new_file = os.path.join(test2_path,file)
        shutil.move(old_file,new_file)
        a = a + 1
        
for file in val_images:
    if file in files:
        old_file = os.path.join(val_path, file)
        new_file = os.path.join(test2_path,file)
        shutil.move(old_file,new_file)
        a = a + 1
        


print(a)
#%%
import pandas as pd
data = pd.read_csv('Data/fmc_d_annotated_images1.csv')
data = data.drop(labels = 'Unnamed: 0', axis = 1)
data = data.sort_values(['set', 'leaf_type', 'leaf_number','drying_stage','spectral_number'])

data.to_csv('Data/fmc_d_annotated_images1.csv', index=False)

#%%

dataa = pd.read_csv('Data/fmc_d_annotated_images1.csv')
