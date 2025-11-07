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

from sklearn.pipeline import Pipeline
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

import seaborn as sns


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

features = [*mean_features, *med_features, *mode_features]

train = pd.read_csv('Data/train_data.csv')
test = pd.read_csv('Data/test_data.csv')

x_train = train.loc[:, features]
x_test = train.loc[:, features]

models_folder = 'Models/no_FMC_20'

models = os.listdir(models_folder)

params_list=[]
scores_list=[]
vis = []
counts2 = {}
scores_d2 = {}
feature_names = x_test.columns
top_features = []

counts = []

for i, model_name in enumerate(models):
    
    # if i%5 != 0:
    #     continue
    
    model_path = os.path.join(models_folder, model_name)
    
    vi = model_name.split('_')[0]
    regressor = model_name.split('_')[1].split('.')[0]
    
    #print(vi)
    
    vis.append(vi)
    
    model = load(model_path)
    
    best_model = model.best_estimator_
    best_params = model.best_params_
    
    selected_mask = best_model.named_steps['select'].get_support()
    selected_indices = best_model.named_steps['select'].get_support(indices = True)
    scores = best_model.named_steps['select'].scores_
    pvalues = best_model.named_steps['select'].pvalues_
    scores_list.append(scores)
     
    
    feature_names = x_test.columns
    selected_features = feature_names[selected_mask]
    
    # if ('FMC_d' or 'FMC_f') not in selected_features:
    #     print(model_name)
    
    # for feature, selected, score in zip(feature_names, selected_mask, scores):
    #     if selected:
    #         if feature in counts2:
    #             counts2[feature] = counts2[feature] + 1
    #             scores_d2[feature] = scores_d2[feature] + score
                
    #         else:
    #             counts2[feature] = 1
    #             scores_d2[feature] = score
    
    
    feature_importance = pd.DataFrame({
        'Feature' : feature_names,
        'Score' : scores,
        'p_value': pvalues,
        'selected': selected_mask
        
        }).sort_values(by = 'Score', ascending = False)
    
    top_feature = feature_importance.iloc[0]['Feature']
    
    select_k = best_params['select__k']
    counts.append(select_k)
    
    top_features.append(top_feature)
    
    ff = f'the top feature for {model_name} is {top_feature} '
    ff2 = f'the number of selected features is : {select_k}'
    print(ff)
    print(ff2)
    
    
counts = Counter(counts)
print(counts)

print(Counter(top_features))

# scores_array = np.array(scores_list)

# A = scores_array.T

# A_norm = (A - A.min(axis=0)) / (A.max(axis=0) - A.min(axis=0))

# score_df = pd.DataFrame(A_norm, index=feature_names, columns=vis)

# plt.figure(figsize=(12, 10))
# sns.heatmap(score_df, cmap='viridis', cbar_kws={'label': 'Score'})
# plt.title('Feature Scores for Each Target')
# plt.xlabel('Targets')
# plt.ylabel('Features')
# plt.tight_layout()
# plt.show()








    
    #params_list.append(best_params['select__k'])
        

        

    
   
    
    
    
    
    
    