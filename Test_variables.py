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

features_vis = [*mean_features, *med_features, *mode_features]

features_vis_texture = [*mean_features, *med_features, *mode_features,
                        *blue_textures, *red_textures, *green_textures,
                        *re_textures, *nir_textures]

features_vis_texture_lwc = ['FMC_d',*mean_features, *med_features, *mode_features,
                            *blue_textures, *red_textures, *green_textures,
                            *re_textures, *nir_textures]

features = features_vis



train = pd.read_csv('Data/train_total.csv')
test = pd.read_csv('Data/test_total.csv')

x_train = train.loc[:, features]
x_test = train.loc[:, features]

models_folder = 'Models/vis/_20'

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
    
    if i%5 != 0:
        continue
    
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
    
    top_feature = feature_importance.iloc[0:5]['Feature']
    top_feature = top_feature.to_list()
        
    select_k = best_params['select__k']
    counts.append(select_k)
    
    top_features.extend(top_feature)
    
    # ff = f'the top feature for {model_name} is {top_feature} '
    # ff2 = f'the number of selected features is : {select_k}'
    # print(ff)
    # print(ff2)
    
#%%
    
# counts = Counter(counts)
# print(counts)
counts = dict(Counter(top_features))

other_sum = sum(value for value in counts.values() if value ==1)
counts2 = {k:v for k,v in counts.items() if v > 1}
counts2["other"] = other_sum



counts2df = pd.DataFrame(list(counts2.items()), columns=['Feature', 'Count'])
counts2_sorted = counts2df.sort_values(by='Count', ascending=False)

Features = [feature.upper().replace('_',' ') for feature in counts2_sorted['Feature']]

try:
    index_to_replace = Features.index('FMC D')
    Features[index_to_replace] = 'LWC'
    
except ValueError:
    print("There is no FMC")
    
try:
    index_to_replace = Features.index('OTHER')
    Features[index_to_replace] = 'OTHER*'
    
except ValueError:
    print("There is no FMC")

fig, ax = plt.subplots(1,1,figsize=(4.4,3))
ax.bar(Features, counts2_sorted['Count'])
ax.set_title('Most used features using VIs')
ax.tick_params(axis = 'x', rotation = 90)
ax.grid(linestyle='--', alpha = 0.7)
fig.tight_layout()
fig.savefig('Paper/Figures/features_vis.pdf')


#%%


print(counts2)

counts3 = { 'MEAN RE': 19,
           'CORRELATION RE': 18,
           'STD RE': 18,
           'VARIANCE RE': 17,
           'MEAN G': 13,
           'CONTRAST B': 6,
           'CORRELATION NIR': 6,
           'CORRELATION G': 5,
           'CORRELATION R': 5,
           'DISSIMILARITY R': 4,
           'DISSIMILARITY B': 4,
           'CONTRAST R': 3,
           'MEAN R': 2,
           'ENTROPY RE': 2,
           'ENTROPY R': 2,
           'ENTROPY NIR': 1,
            }

counts3df = pd.DataFrame(list(counts3.items()), columns=['Feature', 'Count'])
counts3_sorted = counts3df.sort_values(by='Count', ascending=False)

Features = [feature.upper().replace('_',' ') for feature in counts3_sorted['Feature']]

fig, ax = plt.subplots(1,1,figsize=(4.4,3))
ax.bar(Features, counts3_sorted['Count'])
ax.set_title('Most used features using texture')
ax.tick_params(axis = 'x', rotation = 90)
ax.grid(linestyle='--', alpha = 0.7)
fig.tight_layout()
fig.savefig('Paper/Figures/features_texture.pdf')


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
        

        

    
   
    
    
    
    
    
    