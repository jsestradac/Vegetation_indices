# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 13:23:26 2025

@author: Robotics
"""

import pandas as pd
import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt 
import os
from utils import get_descriptors, add_padding

data = pd.read_csv('Data_with_descriptors.csv')

features = ['ndvi', 'gndvi', 'ndre', 'sipi','ngbdi', 'ngrdi',
            'grdi', 'nbgvi', 'negi', 'mgrvi', 'mvari', 'rgbvi',
            'tgi', 'vari', 'grri', 'nri', 'grvi', 'sr', 'savi',
            'cl', 'psri', 'm3cl', 'si', 'msr', 'osavi','rvi',
            'rvi2', 'tvi', 'evi', 'gi', 'tcari', 'srpi', 'npci',
            'ndvigb', 'psri2', 'cive', 'nirv', 'dvi', 'msavi',
            'cari', 'remsr', 'rendvi', 'lci', 'b', 'g', 'r', 'nir', 're' ]



blue_images = data['blue'].to_list()
green_images = data['green'].to_list()
red_images = data['red'].to_list()
nir_images = data['nir'].to_list()
re_images = data['red_edge'].to_list()

b_mean_list = []
g_mean_list = []
r_mean_list = []
nir_mean_list = []
re_mean_list = []

b_med_list = []
g_med_list = []
r_med_list = []
nir_med_list = []
re_med_list = []

b_mode_list = []
g_mode_list = []
r_mode_list = []
nir_mode_list = []
re_mode_list = []

for path_b, path_g, path_r, path_nir, path_re in zip(blue_images, green_images,
                                                     red_images, nir_images,
                                                     re_images):
    img_b = cv.imread(path_b, -1)
    img_g = cv.imread(path_g, -1)
    img_r = cv.imread(path_r, -1)
    img_nir = cv.imread(path_nir, -1)
    img_re = cv.imread(path_re, -1)
    
    img_b = img_b / (2**16-1)
    img_g = img_g / (2**16-1)
    img_r = img_r / (2**16-1)
    img_nir = img_nir / (2**16-1)
    img_re = img_re / (2**16-1)
    
    img_b = add_padding(img_b)
    img_g = add_padding(img_g)
    img_r = add_padding(img_r)
    img_nir = add_padding(img_nir)
    img_re = add_padding(img_re)
    
    
    b_mean, b_med, b_mode = get_descriptors(img_b)
    g_mean, g_med, g_mode = get_descriptors(img_g)
    r_mean, r_med, r_mode = get_descriptors(img_r)
    nir_mean, nir_med, nir_mode = get_descriptors(img_nir)
    re_mean, re_med, re_mode = get_descriptors(img_re)
    
    b_mean_list.append(b_mean)
    g_mean_list.append(g_mean)
    r_mean_list.append(r_mean)
    nir_mean_list.append(nir_mean)
    re_mean_list.append(re_mean)

    b_med_list.append(b_med)
    g_med_list.append(g_med)
    r_med_list.append(r_med)
    nir_med_list.append(nir_med)
    re_med_list.append(re_med)

    b_mode_list.append(b_mode)
    g_mode_list.append(g_mode)
    r_mode_list.append(r_mode)
    nir_mode_list.append(nir_mode)
    re_mode_list.append(re_mode)
#%%   
new_features = {'b_mean': b_mean_list,
                'g_mean': g_mean_list,
                'r_mean': r_mean_list,
                'nir_mean': nir_mean_list,
                're_mean': re_mean_list,
                'b_med': b_med_list,
                'g_med': g_med_list,
                'r_med': r_med_list,
                'nir_med': nir_med_list,
                're_med': re_med_list,
                'b_mode': b_mode_list,
                'g_mode': g_mode_list,
                'r_mode': r_mode_list,
                'nir_mode': nir_mode_list,
                're_mode': re_mode_list}

df = pd.DataFrame(new_features)
new_dataset = pd.concat([data, df], axis = 1)
new_dataset.to_csv('Data_with_vi_and_band_descriptors.csv', index = False)
    
#%%
final_data = pd.read_csv('Data_with_vi_and_band_descriptors.csv')
    
    