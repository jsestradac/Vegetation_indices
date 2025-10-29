# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:47:30 2025

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

models_path = 'Models/K_best'

models_list = os.listdir(models_path)

if not models_list:
    sys.exit('Error in the models folder')
    

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

for model_name in tqdm(models_list, desc = 'progress', total = len(models_list)):
    
    model = load(os.path.join(models_path, model_name))
    
    
    best_model = model.best_estimator_
    
    vi = model_name.split('_')[0]
    
    y_train = train[vi]
    
    y_test = test[vi]
    y_test_avo = test_avo[vi]
    y_test_olive = test_olive[vi]
    y_test_grape = test_grape[vi]
    
    y_pred = best_model.predict(x_test)
    y_pred_avo = best_model.predict(x_test_avo)
    y_pred_olive = best_model.predict(x_test_avo)
    y_pred_grape = best_model.predict(x_test_grape)
    
    y_pred_tr = best_model.predict(x_train)
    
    r2_tr = r2_score(y_train, y_pred_tr)
    r2 = r2_score(y_pred, y_test)
    r2_avo = r2_score(y_test_avo, y_pred_avo)
    r2_olive = r2_score(y_test_olive, y_pred_olive)
    r2_grape = r2_score(y_test_grape, y_pred_grape)
    
    rmse_tr = root_mean_squared_error(y_train, y_pred_tr)
    rmse = root_mean_squared_error(y_pred, y_test)
    rmse_avo = root_mean_squared_error(y_test_avo, y_pred_avo)
    rmse_olive = root_mean_squared_error(y_test_olive, y_pred_olive)
    rmse_grape = root_mean_squared_error(y_test_grape, y_pred_grape)
    
    mae_tr = mean_absolute_error(y_train, y_pred_tr)
    mae = mean_absolute_error(y_pred, y_test)
    mae_avo = mean_absolute_error(y_test_avo, y_pred_avo)
    mae_olive = mean_absolute_error(y_test_olive, y_pred_olive)
    mae_grape = mean_absolute_error(y_test_grape, y_pred_grape)
    
    y_min = np.amin(np.concatenate([y_test, y_pred]))
    #y_min = np.floor(y_min)
    
    y_max = np.amax(np.concatenate([y_test, y_pred]))
    #y_max = np.ceil(y_max)
    
    x_vector = np.linspace(0.95*y_min, 1.05*y_max,100)
    
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
    
    list_param.append(str(model.best_params_))
    list_models.append(model_name)
    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(y_test, y_pred, marker = 'o', linestyle = '', 
                 color = 'red', markersize = 1,)
    ax[0,0].plot(x_vector, x_vector)
    ax[0,0].grid()
    ax[0,0].set_title('Complete Dataset')
    ax[0,0].set_xlabel('Predicted Values')
    ax[0,0].set_ylabel('Real Values')
    ax[0,0].text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
                 transform=ax[0,0].transAxes, fontsize=8,
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    ax[0,1].plot(y_test_avo, y_pred_avo, marker = 'o', linestyle = '', 
                 color = 'red', markersize = 1,)
    ax[0,1].plot(x_vector, x_vector)
    ax[0,1].grid()
    ax[0,1].set_title('Avocado')
    ax[0,1].set_xlabel('Predicted Values')
    ax[0,1].set_ylabel('Real Values')
    ax[0,1].text(0.05, 0.95, f'$R^2$ = {r2_avo:.3f}\nRMSE = {rmse_avo:.3f}\nMAE = {mae_avo:.3f}', 
                 transform=ax[0,1].transAxes, fontsize=8,
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    ax[1,0].plot(y_test_olive, y_pred_olive, marker = 'o', linestyle = '', 
                 color='red', markersize = 1,)
    ax[1,0].plot(x_vector, x_vector)
    ax[1,0].grid()
    ax[1,0].set_title('Olive')
    ax[1,0].set_xlabel('Predicted Values')
    ax[1,0].set_ylabel('Real Values')
    ax[1,0].text(0.05, 0.95, f'$R^2 $= {r2_olive:.3f}\nRMSE = {rmse_olive:.3f}\nMAE = {mae_olive:.3f}', 
                 transform=ax[1,0].transAxes, fontsize=8,
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    
    ax[1,1].plot(y_test_grape, y_pred_grape, marker = 'o', linestyle = '', 
                 color = 'red', markersize = 1,)
    ax[1,1].plot(x_vector, x_vector)
    ax[1,1].grid()
    ax[1,1].set_title('Grape')
    ax[1,1].set_xlabel('Predicted Values')
    ax[1,1].set_ylabel('Real Values')
    ax[1,1].text(0.05, 0.95, f'$R^2$ = {r2_grape:.3f}\nRMSE = {rmse_grape:.3f}\nMAE = {mae_grape:.3f}', 
                 transform=ax[1,1].transAxes, fontsize=8,
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    regressor = model_name.split('_')[1].split('.')[0]
    
    plot_title = regressor + ' Results for: ' + vi
    
    fig.suptitle(plot_title)
    fig.tight_layout()
    plt.show()
    
    #model_path = 'Models/' + target + '_' + model_name + '.pkl'
    figure_path = 'Figures/' + vi + '_' + regressor + '.pdf'
    figure_path2 = 'Figures/' + vi + '_' + regressor + '.png'
    # fig.savefig(figure_path)
    # fig.savefig(figure_path2)
    
    
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
    
    plot_title = regressor + ' Results for: ' + vi + ' (Training) '
    #figure_path = 'Figures/SVR_' + target + '_training.pdf'
    figure_path = 'Figures/' + vi + '_' + regressor + '_training.png'
    figure_path2 = 'Figures/' + vi + '_' + regressor + '_training.pdf'
    fig2.suptitle(plot_title)
    # fig2.savefig(figure_path)
    # fig2.savefig(figure_path2)
    
    
    
    
    
    fig3, ax = plt.subplots(1,1)
    plot_title = regressor + ' Results for: ' + vi 
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
    
    new_folder_path = os.path.join('Figures','Full_data')
    
    if not os.path.isdir(new_folder_path):
        os.mkdir(new_folder_path)
    
    
    name_figure_pdf = vi + '_' + regressor + '_full_data.pdf'
    name_figure_png = vi + '_' + regressor + '_full_data.png'
    
    
    figure_path = os.path.join(new_folder_path, name_figure_pdf)
    figure_path2 = os.path.join(new_folder_path, name_figure_png)
    
    fig3.tight_layout()
    # fig3.savefig(figure_path)
    # fig3.savefig(figure_path2)
    
    
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
#metrics_sorted = metrics_df.sort_values(by = 'model')
metrics_df.to_csv('metrics_recreated_k_best.csv', index='False')


#%%

# metrics = pd.read_csv('metrics.csv')
metrics_sorted = metrics_df.sort_values(by = 'model')
metrics_sorted.to_csv('metrics_sorted_k_best.csv', index ='False')

#%% 

metrics_recreated = pd.read_csv('metrics_recreated_k_best.csv')
metrics_recreated = metrics_recreated.sort_values(by = 'model')   

#%%

    
    