# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 13:11:50 2025

@author: Robotics
"""

import os 
from joblib import load
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

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

features_vec = [features_fmc_vis, 
                features_vis, 
                features_vis_texture,
                features_texture, 
                features_FMC_vis_texture]

models_folders = ['fmc_vis',
                  'vis',
                  'vis_texture',
                  'texture',
                  'vis_texture_fmc']

metrics_20 = pd.read_csv('Metrics/metrics_vis_FMC_texture_20k.csv')
metrics_80 = pd.read_csv('Metrics/metrics_vis_FMC_texture_80k.csv')
metrics_195 = pd.read_csv('Metrics/metrics_vis_FMC_texture_195k.csv')

metrics_20['file'] = 'Models/vis_texture_fmc/_20/' + metrics_20['model'] + '.pkl'
metrics_80['file'] = 'Models/vis_texture_fmc/_80/' + metrics_80['model'] + '.pkl'
metrics_195['file'] = 'Models/vis_texture_fmc/_195/'  + metrics_195['model'] + '.pkl'

metrics_20['input'] = 20
metrics_80['input'] = 80
metrics_195['input'] = 195

metrics_concatenated = pd.concat([metrics_20, metrics_80, metrics_195], ignore_index=True)

idx_max_r2 = metrics_concatenated.groupby('model')['r2'].idxmax()

best_models = metrics_concatenated.loc[idx_max_r2]

best_models = best_models.loc[:,['model','mae','rmse','r2','file','input']]
#%%
VIs = best_models['model'].str.split('_').str[0]
ML_model = best_models['model'].str.split('_').str[1]

best_models['VI'] = VIs
best_models['ML_model'] = ML_model


#%%

idx_models_by_VI = best_models.groupby('VI')['r2'].idxmax()

best_models_by_VI = best_models.loc[idx_models_by_VI]

best_models_by_VI.to_csv('Metrics/best_models_with_lwc.csv', index = False)

#%%





