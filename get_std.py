# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:34:35 2025

@author: Robotics
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import sys
import os 
from joblib import load 
from tqdm import tqdm


new_metrics_path = 'Metrics/vis_std.csv'


data = pd.read_csv('Metrics/a_vis.csv')





#%%

folders = {'lwc_vis_texture': 'Models/vis_texture_fmc/',
           'texture': 'Models/texture/',
           'vis_texture': 'Models/vis_texture/',
           'vis': 'Models/fmc_vis/'}


def get_folder(x):
    numb = x.split('k')[0]
    return '_'+numb+'/'

file_folder =  data['data_input'].apply(lambda x: folders[x]) + data['features'].apply(get_folder) + data['model'] + '.pkl'

data['file'] = file_folder

vi_features = ['ndvi', 'gndvi', 'ndre', 'sipi','ngbdi', 'ngrdi',
            'grdi', 'nbgvi', 'negi', 'mgrvi', 'mvari', 'rgbvi',
            'tgi', 'vari', 'grri', 'nri', 'grvi', 'sr', 'savi',
            'cl', 'psri', 'm3cl', 'si', 'msr', 'osavi','rvi',
            'rvi2', 'tvi', 'evi', 'gi', 'tcari', 'srpi', 'npci',
            'ndvigb', 'psri2', 'cive', 'nirv', 'dvi', 'msavi',
            'cari', 'remsr', 'rendvi', 'lci', 'b', 'g', 'r', 'nir', 're' ]


targets = ['WBI', 'MSI', 'MSI1', 'MSI2',
           'TM57', 'WI', 'FWBI', 'LWI', 
           'SWRI', 'SWRI1', 'SRWI2', 'NDWI',
           'NDWI1', 'NDWI2', 'SIWSI', 'DDI',
           'MAX', 'VOG1', 'SPADI', 'PSRI', 
           'RVSI', 'NDVI1', 'SIPI', 'LIC2',
           'DD']

texture_features = ['ASM', 'contrast', 'correlation', 'dissimilarity', 'energy',
                     'entropy', 'homogeneity', 'mean', 'std', 'variance']

blue_textures = [texture_feature +'_B' for texture_feature in texture_features]
red_textures = [texture_feature +'_R' for texture_feature in texture_features]
green_textures = [texture_feature +'_G' for texture_feature in texture_features]
re_textures = [texture_feature +'_RE' for texture_feature in texture_features]
nir_textures = [texture_feature +'_NIR' for texture_feature in texture_features]

mean_features = [feature + '_mean' for feature in vi_features]
med_features = [feature + '_med' for feature in vi_features]
mode_features = [feature +'_mode' for feature in vi_features]



features_fmc_vis = ['FMC_f', 'FMC_d', 'chlorophyll',*mean_features, *med_features,
                 *mode_features]

features_vis = [*mean_features, *med_features, *mode_features]

features_vis_texture = [*mean_features, *med_features, *mode_features,
                       *blue_textures, *red_textures, *green_textures,
                       *re_textures, *nir_textures]

features_texture = [*blue_textures, *red_textures, *green_textures,
                    *re_textures, *nir_textures]

features_FMC_vis_texture = ['FMC_d',*mean_features, *med_features, *mode_features,
                            *blue_textures, *red_textures, *green_textures,
                            *re_textures, *nir_textures]

features = features_fmc_vis.copy()

list_r2 = []
list_r2_avo = []
list_r2_olive = []
list_r2_grape = []

list_rmse = []
list_rmse_avo = []
list_rmse_olive = []
list_rmse_grape = []

list_mae = []
list_mae_avo = []
list_mae_olive = []
list_mae_grape = []

list_param = []
list_models = []

list_r2_tr = []
list_mae_tr = []
list_rmse_tr = []

list_models = []

train = pd.read_csv('Data/train_total.csv')
test = pd.read_csv('Data/test_total.csv')

x_train = train.loc[:, features]
x_test = test.loc[:, features]

test_avo = test[test['Species'] == 'Avocado']
x_test_avo = test_avo.loc[:,features]

test_olive = test[test['Species'] == 'olive']
x_test_olive = test_olive.loc[:,features]

test_grape = test[test['Species'] == 'vineyard']
x_test_grape = test_grape.loc[:,features]

for index, row in tqdm(data.iterrows(), total = 95):
    
    model = load(row['file'])
    best_model = model.best_estimator_
    
    vi = row['VI']
    if vi == 'SWRI2':
        vi = 'SRWI2'
    
    y_train = train[vi]
    
    y_test = test[vi]
    y_test_avo = test_avo[vi]
    y_test_olive = test_olive[vi]
    y_test_grape = test_grape[vi]
    
    y_pred = best_model.predict(x_test)
    y_pred_avo = best_model.predict(x_test_avo)
    y_pred_olive = best_model.predict(x_test_avo)
    y_pred_grape = best_model.predict(x_test_grape)
    
    err = y_test - y_pred
    err = err.to_numpy()
    
    mae_std = np.std(err)
    data.loc[index,'mae_std'] = mae_std
    

    
final_metrics = data[['ML_model','VI','mae','mae_std','r2',]]

final_metrics.to_csv(new_metrics_path, index=False)
    
    
    
    
    

