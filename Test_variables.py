# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 16:45:27 2025

@author: Robotics
"""


import pandas as pd 
import sys
import os 
import numpy as np 
import matplotlib.pyplot as plt
from joblib import load
from collections import Counter
from tqdm import tqdm

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

mean_features = [feature + '_mean' for feature in vi_features]
med_features = [feature + '_med' for feature in vi_features]
mode_features = [feature +'_mode' for feature in vi_features]

features = ['FMC_f', 'FMC_d', 'chlorophyll',*mean_features, *med_features,
                  *mode_features]

train = pd.read_csv('Data/train_data.csv')
test = pd.read_csv('Data/test_data.csv')

x_train = train.loc[:, features]

models_folder = 'Models/_50'

models = os.listdir(models_folder)

params_list=[]

for model_name in tqdm(models):
    
    model_path = os.path.join(models_folder, model_name)
    
    vi = model_name.split('_')[0]
    regressor = model_name.split('_')[1].split('.')[0]
    
    model = load(model_path)
    
    best_model = model.best_estimator_
    
    best_params = model.best_params_
    
    
    params_list.append(best_params['select__k'])
        
counts = Counter(params_list)
print('\n')
print(counts)
        

    
   
    
    
    
    
    
    