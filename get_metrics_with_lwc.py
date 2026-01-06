# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:24:07 2025

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
from utils import *

configure_plots()

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

features = features_FMC_vis_texture.copy()
#%%

figure_folder = os.path.join('Figures','LWC_Real')
if not os.path.isdir(figure_folder):
     os.mkdir(figure_folder)



metrics_path = 'Metrics/metrics_lwc_real.csv'


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

list_mae_std = []


models = pd.read_csv('metrics/best_models_with_lwc.csv')
models_list = models['file'].to_list()


test = pd.read_csv('Data/total_test_ordered.csv')



lwc_resnet = pd.read_csv('Data/pred_lwc_resnet.csv')
lwc_mobilenet = pd.read_csv('Data/pred_lwc_mobilenet.csv')

lwc_real = test['FMC_d']

lwc = pd.concat([lwc_real, lwc_mobilenet, lwc_resnet], ignore_index=True)


lwc_resnet_arr = lwc_resnet['predicted lwc'].to_list()
lwc_mobilenet_arr = lwc_mobilenet['predicted lwc'].to_list()

lwc_resnet_down = [x for (i,x) in enumerate(lwc_resnet_arr) if i % 5 == 0]
lwc_mobilent_down = [x for (i,x) in enumerate(lwc_mobilenet_arr) if i % 5 == 0]



if not models_list:
    sys.exit('Error in the models folder')
    

train = pd.read_csv('Data/train_total.csv')



x_test = test.loc[:, features]
x_train = train.loc[:,features]

test_avo = test[test['Species'] == 'Avocado']
x_test_avo = test_avo.loc[:,features]

test_olive = test[test['Species'] == 'olive']
x_test_olive = test_olive.loc[:,features]

test_grape = test[test['Species'] == 'vineyard']
x_test_grape = test_grape.loc[:,features]


x_test.loc[:,'FMC_d'] = lwc_real
#x_test.loc[:,'FMC_d'] = x_test['FMC_d'].apply(lambda x: x +np.random.uniform(0,24))

metrics_path = 'Metrics/real_lwc.csv'
#%%


for model_name in tqdm(models_list, desc = 'progress', total = len(models_list)):
    
    model = load(model_name)
    
    model_str = os.path.basename(model_name)
    
    
    best_model = model.best_estimator_
    
    # x_test.loc[:,'FMC_d'] = x_test['FMC_d'].apply(lambda x: x +np.random.uniform(0,11))
    
    

    
    vi = model_str.split('_')[0]
    ml_model = model_str.split('_')[1].split('.')[0]
    
    y_train = train[vi]
    
    y_test = test[vi]
    y_test_avo = test_avo[vi]
    y_test_olive = test_olive[vi]
    y_test_grape = test_grape[vi]
#%%
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
    
    err = y_pred - y_test
    
    err_np = err.to_numpy()
    
    mae_std = np.std(err)
    
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
    list_mae_std.append(mae_std)
    
    # fig, ax = plt.subplots(2, 2, figsize=(4.3,2.33))
    # ax[0,0].plot(y_test, y_pred, marker = 'o', linestyle = '', 
    #              color = 'red', markersize = 1,)
    # ax[0,0].plot(x_vector, x_vector)
    # ax[0,0].grid()
    # ax[0,0].set_title('Complete Dataset')
    # ax[0,0].set_xlabel('Predicted Values')
    # ax[0,0].set_ylabel('Real Values')
    # ax[0,0].text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
    #              transform=ax[0,0].transAxes, fontsize=8,
    #              verticalalignment='top', 
    #              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # ax[0,1].plot(y_test_avo, y_pred_avo, marker = 'o', linestyle = '', 
    #              color = 'red', markersize = 1,)
    # ax[0,1].plot(x_vector, x_vector)
    # ax[0,1].grid()
    # ax[0,1].set_title('Avocado')
    # ax[0,1].set_xlabel('Predicted Values')
    # ax[0,1].set_ylabel('Real Values')
    # ax[0,1].text(0.05, 0.95, f'$R^2$ = {r2_avo:.3f}\nRMSE = {rmse_avo:.3f}\nMAE = {mae_avo:.3f}', 
    #              transform=ax[0,1].transAxes, fontsize=8,
    #              verticalalignment='top', 
    #              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # ax[1,0].plot(y_test_olive, y_pred_olive, marker = 'o', linestyle = '', 
    #              color='red', markersize = 1,)
    # ax[1,0].plot(x_vector, x_vector)
    # ax[1,0].grid()
    # ax[1,0].set_title('Olive')
    # ax[1,0].set_xlabel('Predicted Values')
    # ax[1,0].set_ylabel('Real Values')
    # ax[1,0].text(0.05, 0.95, f'$R^2 $= {r2_olive:.3f}\nRMSE = {rmse_olive:.3f}\nMAE = {mae_olive:.3f}', 
    #              transform=ax[1,0].transAxes, fontsize=8,
    #              verticalalignment='top', 
    #              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    
    # ax[1,1].plot(y_test_grape, y_pred_grape, marker = 'o', linestyle = '', 
    #              color = 'red', markersize = 1,)
    # ax[1,1].plot(x_vector, x_vector)
    # ax[1,1].grid()
    # ax[1,1].set_title('Grape')
    # ax[1,1].set_xlabel('Predicted Values')
    # ax[1,1].set_ylabel('Real Values')
    # ax[1,1].text(0.05, 0.95, f'$R^2$ = {r2_grape:.3f}\nRMSE = {rmse_grape:.3f}\nMAE = {mae_grape:.3f}', 
    #              transform=ax[1,1].transAxes, fontsize=8,
    #              verticalalignment='top', 
    #              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # regressor = 'Real LWC '
    
    # plot_title = regressor + ' Results for: ' + vi
    
    # fig.suptitle(plot_title)
    # fig.tight_layout()
    
    # figure_name_pdf = vi + '_' + regressor + '.pdf'
    # figure_name_png = vi + '_' + regressor + '.png'
    
    # figure_path_pdf = os.path.join(figure_folder, figure_name_pdf)
    # figure_path_png = os.path.join(figure_folder, figure_name_png)
    
    # # fig.savefig(figure_path_pdf)
    # # fig.savefig(figure_path_png)
    
    
    # fig2, ax = plt.subplots(1,1, figsize = (4.31,2.3))
    # ax.plot(y_train, y_pred_tr, marker = 'o', linestyle = '', 
    #              color='red', markersize = 1,)
    # plot_title = regressor + ' Results for: ' + vi + ' (Training) '
    # ax.plot(x_vector, x_vector)
    # ax.grid()
    # ax.set_title(plot_title)
    # ax.set_xlabel('Predicted Values')
    # ax.set_ylabel('Real Values')
    # ax.text(0.05, 0.95, f'$R^2$ = {r2_tr:.3f}\nRMSE = {rmse_tr:.3f}\nMAE = {mae_tr:.3f}$', 
    #              transform=ax.transAxes, fontsize=8,
    #              verticalalignment='top', 
    #              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    
   
    # figure_name_pdf = vi + '_' + regressor + '_training.pdf'
    # figure_name_png = vi + '_' + regressor + '_training.png'
    
    # figure_path_pdf = os.path.join(figure_folder, figure_name_pdf)
    # figure_path_png = os.path.join(figure_folder, figure_name_png)
    
    # fig2.savefig(figure_path_pdf)
    # fig2.savefig(figure_path_png)
    
    
    # fig3, ax = plt.subplots(1,1,figsize=(4.31,2.34))
    # plot_title = regressor + ' Results for: ' + vi 
    # ax.plot(y_test, y_pred, marker = 'o', linestyle = '', 
    #              color = 'red', markersize = 3,)
    # ax.plot(x_vector, x_vector)
    # ax.grid()
    # ax.set_title(plot_title)
    # ax.set_xlabel('Predicted Values')
    # ax.set_ylabel('Real Values')
    # ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
    #              transform=ax.transAxes, fontsize=8,
    #              verticalalignment='top', 
    #              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # new_folder_path = os.path.join(figure_folder,'Full_data')
    
    # if not os.path.isdir(new_folder_path):
    #     os.mkdir(new_folder_path)
    
    
    # name_figure_pdf = vi + '_' + regressor + '_full_data.pdf'
    # name_figure_png = vi + '_' + regressor + '_full_data.png'
    
    
    # figure_path_pdf = os.path.join(new_folder_path, name_figure_pdf)
    # figure_path_png = os.path.join(new_folder_path, name_figure_png)
    
    # fig3.tight_layout()
    # fig3.savefig(figure_path_pdf)
    # fig3.savefig(figure_path_png)
    
    # plt.close()
    
    
metrics = {'model': list_models,
           'mae': list_mae,
           'mae_std': list_mae_std,
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

metrics_pd = pd.DataFrame(metrics)
metrics_pd.to_csv(metrics_path)


