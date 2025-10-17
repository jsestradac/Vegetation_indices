# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 18:25:10 2025

Pipeline for machine_learning
@author: Robotics
"""

import pandas as pd
import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt 
import os
from utils import get_descriptors, add_padding

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRFRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes



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

total_features = ['FMC_f', 'FMC_d', 'chlorophyll',*mean_features, *med_features,
                  *mode_features]

#%%

data = data = pd.read_csv('Data_with_vi_and_band_descriptors.csv')
#%%

if __name__ =='__main__':

    data = pd.read_csv('Data_with_vi_and_band_descriptors.csv')
    
    random_data = data.sample(frac=1)
    
    #%%
    data = pd.read_csv('data_random_order.csv')
    
    
    
    
    
    features_data = data.loc[:, total_features]
    
    #%%
    
    
    
    target_data = data.loc[:,targets]
    
    #%% PREPARE FOR SCIKITLEARN
    
    
    
    x_train, x_test, y_train, y_test = train_test_split(features_data, target_data,
                                                        test_size = 0.2)
    
    scaler = MinMaxScaler()
    x_train_scales = scaler.fit_transform(x_train)
    x_train_scales = pd.DataFrame(x_train_scales, columns=x_train.columns)
    
    
    crossv = KFold(n_splits = 5, shuffle = True, random_state=42)
    
    preprocessing = [
        ('scaler', MinMaxScaler()),
        ('select', SelectKBest(score_func = f_regression, k = 8)),
        ('SVR', SVR())]
    
    ff = x_train
    yy = y_train['LWI']
    
    
    pipeline = Pipeline(preprocessing)
    
    
    score = cross_val_score(pipeline, ff, yy, cv = crossv, scoring = 'r2')
    
    param_grid={
                'SVR__C': [0.1, 1, 10, 100, 1000],
                'SVR__epsilon': [0.0005,  0.001, 0.005,  0.01, 0.05, 1, 5, 10],
                'SVR__gamma': [0.005,  0.01, 0.05, 1, 5, 10],
                'SVR__kernel': ['linear', 'rbf']
            }
    #%%
    grid = GridSearchCV(pipeline, param_grid, cv=crossv, scoring='r2', n_jobs=-1, verbose=3)
    history = grid.fit(ff,yy)

    print("Best parameters:", grid.best_params_)
    print("Best cross-validated R²:", grid.best_score_)
    #%%
    best_model = grid.best_estimator_
    
    from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
    
    avocado_data = data[data['Species']=='Avocado']
    grape_data =  data[data['Species']== 'olive']
    olive_data = data[data['Species'] == 'vineyard' ]
    
    x_avocado = avocado_data.loc[:, total_features]
    y_avocado = avocado_data.loc[:, targets]
    y_avocado = y_avocado['LWI']
    
    x_olive = olive_data.loc[:, total_features]
    y_olive = olive_data.loc[:, targets]
    y_olive = y_olive['LWI']
    
    x_grape = grape_data.loc[:, total_features]
    y_grape = grape_data.loc[:, targets]
    y_grape = y_grape['LWI']
    
    
    
    y_pred_avo = best_model.predict(x_avocado)
    
    
    r2 = r2_score(y_avocado, y_pred_avo)
    rmse = root_mean_squared_error(y_avocado, y_pred_avo)
    mae = mean_absolute_error(y_avocado, y_pred_avo)
    
    print('Avocado')
    print(f"R² : {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    
    y_pred_olive = best_model.predict(x_olive)
    
    r2 = r2_score(y_olive, y_pred_olive)
    rmse = root_mean_squared_error(y_olive, y_pred_olive)
    mae = mean_absolute_error(y_olive, y_pred_olive)
    
    print('OLIVE')
    print(f"R² : {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    
    
    y_pred_vine = best_model.predict(x_grape)
   
    
    r2 = r2_score(y_grape, y_pred_vine)
    rmse = root_mean_squared_error(y_grape, y_pred_vine)
    mae = mean_absolute_error(y_grape, y_pred_vine)
    
    print('GRAPE')
    
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    
    
    y_pred = best_model.predict(ff)
   
    
    r2 = r2_score(yy, y_pred)
    rmse = root_mean_squared_error(yy, y_pred)
    mae = mean_absolute_error(yy, y_pred)
    
    #%%
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(y_grape, y_pred_vine, marker = 'o', linestyle = '', 
                 color=np.random.rand(3,), markersize = 1,)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Leaf Water Content')
    ax[0].grid()
    
    # ax[1].plot(y_avocado, y_pred_avo, marker = 'o', linestyle = '', 
    #              color=np.random.rand(3,), markersize = 1)
    # # ax[0,1].set_xticks([])
    # # ax[0,1].set_yticks([])
    # ax[1].set_title('Leaf Chlorophyll Content')
    
    
    # ax[1,0].plot(y_olive, y_pred_olive, marker = 'o', linestyle = '',
    #              color=np.random.rand(3,), markersize = 1)
    # # ax[1,0].set_xticks([])
    # # ax[1,0].set_yticks([])
    # ax[1,0].set_title('Leaf Nitrogen Content')
    
    ax[1].plot(y_pred, yy, marker = 'o', linestyle = '', 
                 color=np.random.rand(3,), markersize = 1)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Biomass Concentration')
    ax[1].grid()
    
    plt.tight_layout()
    #%%
    np.random.seed(42)

# Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    for ax in axes.flatten():
        # Generate random data
        x = np.arange(5)
        y = np.random.randint(5, 20, size=5)
        
        # Plot bar chart
        ax.bar(x, y, color=np.random.rand(3,))
        
        # Remove all titles, axis labels, and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.axis('off')
    
    
    
    
   # scores = cross_val_score(pipeline, x_train.to_numpy(), y_train['LWI'].to_numpy(), cv=crossv, scoring = 'r2')
    
#%%

    np.random.seed(42)

# Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    for ax in axes.flatten():
        # Generate random scatter data
        x = np.random.rand(50)
        y = np.random.rand(50)
        
        # Random color for each plot
        ax.scatter(x, y, s=50, color=np.random.rand(3,), alpha=0.7)
        
        # Remove all axes, ticks, labels, and frames
        ax.set_xticks([])
        ax.set_yticks([])




