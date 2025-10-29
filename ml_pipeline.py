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
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import joblib
from tqdm import tqdm

#%% Define features and empty lists
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
#%%

if __name__ =='__main__':
    
    #%% Define Model and params
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    
    # data = pd.read_csv('Data_with_vi_and_band_descriptors.csv')
    # train, test, y_train, y_test = train_test_split(data, data,
    #                                                     test_size = 0.2)
    # x_train = train.loc[:, features]
    # x_test = test.loc[:, features]
    
    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')
    
    x_train = train.loc[:, features]
    x_test = test.loc[:, features]
    
    test_avo = test[test['Species'] == 'Avocado']
    x_test_avo = test_avo.loc[:,features]
    
    test_olive = test[test['Species'] == 'olive']
    x_test_olive = test_olive.loc[:,features]
    
    test_grape = test[test['Species'] == 'vineyard']
    x_test_grape = test_grape.loc[:,features]
    
    scaler = MinMaxScaler()
    x_train_scales = scaler.fit_transform(x_train)
    x_train_scales = pd.DataFrame(x_train_scales, columns=x_train.columns)
    crossv = KFold(n_splits = 5, shuffle = True, random_state=42)
    preprocessing_SVR = [
        ('scaler', MinMaxScaler()),
        ('select', SelectKBest(score_func = f_regression, k = 8)),
        ('SVR', SVR())]
    
    pipeline_SVR = Pipeline(preprocessing_SVR)
    param_grid_SVR={
                'select__k': [5, 8, 10, 15 , 20],
                'SVR__C': [0.1, 1, 10, 100, 1000],
                'SVR__epsilon': [0.01, 0.05, 1, 5, 10],
                'SVR__gamma': [0.01, 0.05, 1, 5, 10],
                'SVR__kernel': ['linear', 'rbf']
            }
    
    preprocessing_RF = [
        ('scaler', MinMaxScaler()),
        ('select', SelectKBest(score_func = f_regression, k = 8)),
        ('rf', RandomForestRegressor())]
    
    param_grid_RF = {
                    'select__k': [5, 8, 10, 15 , 20],
                    'rf__n_estimators': [100, 200, 500, 1000 ],           # number of trees
                    'rf__max_depth': [None, 10, 20, 30, 40],          # tree depth
                    'rf__min_samples_split': [2, 5, 7, 10, 15, 20]           # minimum samples to split
                }
    
    pipeline_RF = Pipeline(preprocessing_RF)
    
    preprocessing_mlp = [
        ('scaler', MinMaxScaler()),
        ('select', SelectKBest(score_func = f_regression, k = 8)),
        ('mlp', MLPRegressor(max_iter=1000, random_state=42))]
    
    param_grid_mlp = {
                    'select__k': [5, 8, 10, 15 , 20],                     
                    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)], 
                    'mlp__activation': ['relu', 'tanh'],           
                    'mlp__alpha': [0.0001, 0.001, 0.01],           
                    'mlp__learning_rate_init': [0.001, 0.01]       
                }
    
    pipeline_mlp = Pipeline(preprocessing_mlp)
    
    pipeline_xgb = Pipeline([
        ('scaler', MinMaxScaler()),
        ('select', SelectKBest(score_func=f_regression)),
        ('xgb', XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
        ])
    
    param_grid_xgb = {
                    'select__k': [5, 8, 10, 15 , 20],
                    'xgb__n_estimators': [100, 200, 500, 1000],     # number of trees
                    'xgb__max_depth': [3, 5, 7, 10],         # tree depth
                    'xgb__learning_rate': [0.01, 0.1],   # step size shrinkage
                    'xgb__subsample': [0.2,0.6,0.8, 1.0],        # row sampling
                    'xgb__colsample_bytree': [0.8, 1.0]  # feature sampling
                }
    
    pipeline_ridge = Pipeline([
                                ('scaler', MinMaxScaler()),
                                ('select', SelectKBest(score_func = f_regression, k = 8)),
                                ('ridge', Ridge(random_state=42))
                            ])
    
    param_grid_ridge = { 
                         'select__k': [5, 8, 10, 15 , 20],
                         'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100],   # Regularization strength
                         'ridge__fit_intercept': [True, False],
                         'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
                         'ridge__max_iter':[10, 50, 100, 300, 1000]
                        }
    
    model_names = ['SVR', 'RF', 'MLP', 'XGB', 'Ridge']
    pipelines = [pipeline_SVR, pipeline_RF, pipeline_mlp, pipeline_xgb, pipeline_ridge]
    grids = [param_grid_SVR, param_grid_RF, param_grid_mlp, param_grid_xgb, param_grid_ridge]
    
    model_folder = os.path.join('Models','K_best')
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
        
    figure_folder = os.path.join('Figures', 'k_best')
    if not os.path.isdir(figure_folder):
        os.mkdir(figure_folder)
        
        
    #%% Main loop
    for model_name, pipeline, param_grid in tqdm(zip(model_names, pipelines, grids),
                                                 desc = 'Different Regressors'): 
        
        if not model_name == 'Ridge':
            print(model_name)
            continue
        
        
    
        for target in tqdm(targets, desc = 'Vegetation indices', leave = False):
            
            vi_model = target + '_' + model_name
            print(vi_model)
            list_models.append(vi_model)
                
            y_train = train.loc[:, target]
            y_test = test.loc[:, target]
            y_test_avo = test_avo.loc[:,target]
            y_test_olive = test_olive.loc[:,target]
            
            y_test_grape = test_grape.loc[:,target]
                
            grid = GridSearchCV(pipeline, param_grid, cv=crossv, scoring='r2', n_jobs=-1, verbose=1)
            grid.fit(x_train, y_train)
            
            #print("Best parameters:", grid.best_params_)
            #print("Best cross-validated R²:", grid.best_score_)
            best_model = grid.best_estimator_
            
            y_pred_tr = best_model.predict(x_train)
            r2_tr = r2_score(y_train, y_pred_tr)
            mae_tr = mean_absolute_error(y_train, y_pred_tr)
            rmse_tr = root_mean_squared_error(y_train, y_pred_tr)
            
            y_pred = best_model.predict(x_test)
            y_pred_avo = best_model.predict(x_test_avo)
            y_pred_olive = best_model.predict(x_test_olive)
            y_pred_grape = best_model.predict(x_test_grape)
             
            r2 = r2_score(y_pred, y_test)
            r2_avo = r2_score(y_test_avo, y_pred_avo)
            r2_olive = r2_score(y_test_olive, y_pred_olive)
            r2_grape = r2_score(y_test_grape, y_pred_grape)
            
            rmse = root_mean_squared_error(y_pred, y_test)
            rmse_avo = root_mean_squared_error(y_test_avo, y_pred_avo)
            rmse_olive = root_mean_squared_error(y_test_olive, y_pred_olive)
            rmse_grape = root_mean_squared_error(y_test_grape, y_pred_grape)
            
            mae = mean_absolute_error(y_pred, y_test)
            mae_avo = mean_absolute_error(y_test_avo, y_pred_avo)
            mae_olive = mean_absolute_error(y_test_olive, y_pred_olive)
            mae_grape = mean_absolute_error(y_test_grape, y_pred_grape)
            
            y_min = np.amin(np.concatenate([y_test, y_pred]))
            #y_min = np.floor(y_min)
            
            y_max = np.amax(np.concatenate([y_test, y_pred]))
            #y_max = np.ceil(y_max)
            
            x_vector = np.linspace(0.95*y_min, 1.05*y_max,100)
            
            #%% print results
            # print('\nTRAIN DATA\n')
            # print(f"R² : {r2_tr:.3f}")
            # print(f"RMSE: {rmse_tr:.3f}")
            # print(f"MAE: {mae_tr:.3f}")
            
            # print('\nTEST DATA\n')
            
            # print('Total Dataset')
            # print(f"R² : {r2:.3f}")
            # print(f"RMSE: {rmse:.3f}")
            # print(f"MAE: {mae:.3f}")
            
            # print('\nAvocado')
            # print(f"R² : {r2_avo:.3f}")
            # print(f"RMSE: {rmse_avo:.3f}")
            # print(f"MAE: {mae_avo:.3f}")
            
            # print('\nOlive')
            # print(f"R² : {r2_olive:.3f}")
            # print(f"RMSE: {rmse_olive:.3f}")
            # print(f"MAE: {mae_olive:.3f}")
            
            # print('\nGrape')
            # print(f"R² : {r2_grape:.3f}")
            # print(f"RMSE: {rmse_grape:.3f}")
            # print(f"MAE: {mae_grape:.3f}")
            #%% append results to lists
            
            list_r2.append(r2)
            list_r2_avo.append(r2_avo)
            list_r2_olive.append(r2_olive)
            list_r2_grape.append(r2_grape)
            
            list_mae.append(mae)
            list_mae_avo.append(mae_avo)
            list_mae_olive.append(mae_olive)
            list_mae_grape.append(mae_grape)
            
            list_rmse.append(rmse)
            list_rmse_avo.append(rmse_avo)
            list_rmse_olive.append(rmse_olive)
            list_rmse_grape.append(rmse_grape)
            
            list_r2_tr.append(r2_tr)
            list_mae_tr.append(mae_tr)
            list_rmse_tr.append(rmse_tr)
            
            list_param.append(str(grid.best_params_))
            
            #%% plot results
            
            
            fig, ax = plt.subplots(2, 2)
            ax[0,0].plot(y_test, y_pred, marker = 'o', linestyle = '', 
                         color = 'red', markersize = 1,)
            ax[0,0].plot(x_vector, x_vector)
            ax[0,0].grid()
            ax[0,0].set_title('Leaf Water Content')
            ax[0,0].set_xlabel('Predicted Values')
            ax[0,0].set_ylabel('Real Values')
            ax[0,0].text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}$', 
                         transform=ax[0,0].transAxes, fontsize=8,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            ax[0,1].plot(y_test_avo, y_pred_avo, marker = 'o', linestyle = '', 
                         color = 'red', markersize = 1,)
            ax[0,1].plot(x_vector, x_vector)
            ax[0,1].grid()
            ax[0,1].set_title('Leaf Water Content: Avocado')
            ax[0,1].set_xlabel('Predicted Values')
            ax[0,1].set_ylabel('Real Values')
            ax[0,1].text(0.05, 0.95, f'$R^2$ = {r2_avo:.3f}\nRMSE = {rmse_avo:.3f}\nMAE = {mae_avo:.3f}$', 
                         transform=ax[0,1].transAxes, fontsize=8,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            ax[1,0].plot(y_test_olive, y_pred_olive, marker = 'o', linestyle = '', 
                         color='red', markersize = 1,)
            ax[1,0].plot(x_vector, x_vector)
            ax[1,0].grid()
            ax[1,0].set_title('Leaf Water Content: Olive')
            ax[1,0].set_xlabel('Predicted Values')
            ax[1,0].set_ylabel('Real Values')
            ax[1,0].text(0.05, 0.95, f'$R^2 $= {r2_olive:.3f}\nRMSE = {rmse_olive:.3f}\nMAE = {mae_olive:.3f}$', 
                         transform=ax[1,0].transAxes, fontsize=8,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            
            ax[1,1].plot(y_test_grape, y_pred_grape, marker = 'o', linestyle = '', 
                         color = 'red', markersize = 1,)
            ax[1,1].plot(x_vector, x_vector)
            ax[1,1].grid()
            ax[1,1].set_title('Leaf Water Content: Grape')
            ax[1,1].set_xlabel('Predicted Values')
            ax[1,1].set_ylabel('Real Values')
            ax[1,1].text(0.05, 0.95, f'$R^2$ = {r2_grape:.3f}\nRMSE = {rmse_grape:.3f}\nMAE = {mae_grape:.3f}$', 
                         transform=ax[1,1].transAxes, fontsize=8,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            plot_title = model_name + ' Results for: ' + target
            
            fig.suptitle(plot_title)
            fig.tight_layout()
            plt.show()
            
            model_name_path = target + '_' + model_name + '.pkl'
            figure_name_pdf = target + '_' + model_name + '.pdf'
            figure_name_png = target + '_' + model_name + '.png'
            
            model_path = os.path.join(model_folder, model_name_path)
            
            
            figure_path_pdf = os.path.join(figure_folder, figure_name_pdf)
            figure_path_png = os.path.join(figure_folder, figure_name_png)
            
            
            
            fig.savefig(figure_path_pdf)
            fig.savefig(figure_path_png)
            
            fig2, ax = plt.subplots(1,1)
            ax.plot(y_train, y_pred_tr, marker = 'o', linestyle = '', 
                         color='red', markersize = 1,)
            ax.plot(x_vector, x_vector)
            ax.grid()
            ax.set_title('Leaf Water Content')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Real Values')
            ax.text(0.05, 0.95, f'$R^2$ = {r2_tr:.3f}\nRMSE = {rmse_tr:.3f}\nMAE = {mae_tr:.3f}$', 
                         transform=ax.transAxes, fontsize=8,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            plot_title = model_name + ' Results for: ' + target + ' (Training) '
            #figure_path = 'Figures/SVR_' + target + '_training.pdf'
            figure_name_pdf =  target + '_' + model_name + '_training.pdf'
            figure_name_png =  target + '_' + model_name + '_training.png'
            
            
            figure_path_pdf = os.path.join(figure_folder, figure_name_pdf)
            figure_path_png = os.path.join(figure_folder, figure_name_png)
            
            fig2.suptitle(plot_title)
            fig2.savefig(figure_path_pdf)
            fig2.savefig(figure_path_png)
            
            
            fig3, ax = plt.subplots(1,1)
            plot_title = model_name + ' Results for: ' + target 
            ax.plot(y_test, y_pred, marker = 'o', linestyle = '', 
                         color = 'red', markersize = 3,)
            ax.plot(x_vector, x_vector)
            ax.grid()
            ax.set_title(plot_title)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Real Values')
            ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
                         transform=ax.transAxes, fontsize=8,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            new_folder_path = os.path.join(figure_folder,'Full_data')
            
            if not os.path.isdir(new_folder_path):
                os.mkdir(new_folder_path)
            
            
            name_figure_pdf = target + '_' + model_name + '_full_data.pdf'
            name_figure_png = target + '_' + model_name + '_full_data.png'
            
            
            figure_path = os.path.join(new_folder_path, name_figure_pdf)
            figure_path2 = os.path.join(new_folder_path, name_figure_png)
            
            fig3.tight_layout()
            fig3.savefig(figure_path)
            fig3.savefig(figure_path2)
            
            
            joblib.dump(grid, model_path)
            
#%% save metrics to a csv
            
    metrics = {'model': list_models,
               'mae': list_mae,
               'rmse': list_rmse,
               'r2': list_r2,
               'mae_avo': list_mae_avo,
               'rmse_avo': list_rmse_avo,
               'r2_avo': list_r2_avo,
               'mae_olive': list_mae_olive,
               'rmse_olive': list_rmse_olive,
               'r2_olive': list_r2_olive,
               'mae_grape': list_mae_grape,
               'rmse_grape': list_rmse_grape,
               'r2_grape': list_r2_grape,
               'mae_tr': list_mae_tr,
               'rmse_tr': list_rmse_tr,
               'r2_tr': list_r2_tr,
               'best_params': list_param
                       
            }
    
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('metrics_k_best_.csv', index='False')
    
#%%

metrics_sorted = metrics_df.sort_values(by='model')
    
        
    
  
    






