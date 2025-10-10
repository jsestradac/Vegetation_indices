# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 13:40:51 2025

@author: Robotics
"""
#%% import libraries

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import ast
import cv2 as cv
import os
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from utils import *
from vegetation_indices import *


#%% Get maximum size for the padding

df = pd.read_csv('Dataset_with_images.csv')

blue_images = df['blue']
green_images = df['green']
red_images = df['red']
nir_images = df['nir']
re_images = df['red_edge']

total_images = pd.concat([blue_images, green_images, 
                          red_images, nir_images, re_images], axis = 0)

total_images = total_images.tolist()

max_x, max_y, max_x_path, max_y_path = get_maximum_size(total_images)

#%% Testing the padding of the images

img0 = cv.imread(max_x_path,-1)
img1 = cv.imread(max_y_path,-1)
img2 = cv.imread(total_images[-100],-1)

img_padded0 = add_padding(max_x, max_y, img0)
img_padded1 = add_padding(max_x, max_y, img1)
img_padded2 = add_padding(max_x, max_y, img2)

__, ax = plt.subplots(2,3)

ax[0,0].imshow(img_padded0)
ax[0,0].set_axis_off()
ax[1,0].imshow(img0)
ax[1,0].set_axis_off()

ax[0,1].imshow(img_padded1)
ax[0,1].set_axis_off()
ax[1,1].imshow(img1)
ax[1,1].set_axis_off()

ax[0,2].imshow(img_padded2)
ax[0,2].set_axis_off()
ax[1,2].imshow(img2)
ax[1,2].set_axis_off()

#%% Compute the vegetation indices and get the descriptors

vis = [ndvi, gndvi, ndre, sipi, ngbdi,
       ngrdi, grdi, nbgvi, negi, mgrvi,
       mvari, rgbvi, tgi, vari, grri,
       nri, grvi, sr, savi, cl, psri,
       m3cl, si, msr, osavi, rvi, rvi2,
       tvi, evi, gi, tcari, srpi,
       npci, ndvigb, psri2, cive, nirv,
       dvi, msavi, cari, remsr, rendvi, lci]    

vis_name = ['ndvi', 'gndvi', 'ndre', 'sipi', 'ngbdi',
            'ngrdi', 'grdi', 'nbgvi', 'negi', 'mgrvi',
            'mvari', 'rgbvi', 'tgi', 'vari', 'grri',
            'nri', 'grvi', 'sr', 'savi', 'cl', 'psri',
            'm3cl', 'si', 'msr', 'osavi', 'rvi', 'rvi2',
            'tvi', 'evi', 'gi', 'tcari', 'srpi',
            'npci', 'ndvigb', 'psri2', 'cive', 'nirv',
            'dvi', 'msavi', 'cari', 'remsr', 'rendvi', 'lci']

average = []
median = []
mode = []

vi_directory = []

flag1 = False
flag2 = False
flag3 = False

last_species = 'aa'

for index, row in tqdm(df.iterrows(), total=len(df), desc = 'Total progress:'):
    
    
    species = row['Species']
    
    path_vis_species = os.path.join(species,'Multispectral Images','VIs')
    
    
    # ###
    # ### comment to only do in the first image of the species
    # if species == last_species: 
    #     last_species = species
        
    #     continue
    
    # last_species = species
    # ####
    # ####
        
    
    path_b = row['blue']
    path_g = row['green']
    path_r = row['red']
    path_nir = row['nir']
    path_re = row['red_edge']

    file_name = Path(path_b)
    file_name = file_name.stem   
    file_name = file_name.split('_')[0]
     
    if not os.path.isdir(path_vis_species):
        os.mkdir(path_vis_species)
        
    b = cv.imread(path_b,-1)
    g = cv.imread(path_g,-1)
    r = cv.imread(path_r,-1)
    nir = cv.imread(path_nir,-1)
    re = cv.imread(path_re,-1)
    
    b = add_padding(b, max_x, max_y)
    g = add_padding(g, max_x, max_y)
    r = add_padding(r, max_x, max_y)
    nir = add_padding(nir, max_x, max_y)
    re = add_padding(re, max_x, max_y)
    
    vi_directory_dict = {}
    average_dict = {}
    median_dict = {}
    mode_dict = {}
    

    
    b = b / (2**16-1)
    g = g / (2**16-1)
    r = r / (2**16-1)
    nir = nir / (2**16-1)
    re = re / (2**16-1)
    
    vi_images_path = os.path.join(species,'Multispectral Images','Vis')
    if not os.path.isdir(vi_images_path):
        os.mkdir(vi_images_path)
    
    
    for f, name in tqdm(zip(vis, vis_name), total = len(vis), desc='Processing Vis', leave=False):
        
        vi_img = f(b,g,r,nir,re)
        av, med, mo = get_descriptors(vi_img)
        vi_path = os.path.join(species,'Multispectral Images','Vis',name)
        
        if not os.path.isdir(vi_path):
            os.mkdir(vi_path)
            
        img_vi_path = os.path.join(vi_path,file_name+'.csv')
        
        ####
        #### uncomment to save the vi_images in the folders
        
        #np.savetxt(img_vi_path, vi_img, delimiter=',')
        

        ####
        ####

        vi_directory_dict[name] = img_vi_path
        average_dict[name+'_mean'] = av
        median_dict[name + '_med'] = med
        mode_dict[name + '_mode'] = mo
        
        
        
        
    average.append(average_dict)
    median.append(median_dict)
    mode.append(mode_dict)
    vi_directory.append(vi_directory_dict)
    
    
    
    
    
#%% Concatenate the dataframes

df2 = pd.DataFrame(average)
df3 = pd.DataFrame(median)
df4 = pd.DataFrame(mode)

df_with_vis_descriptors = pd.concat([df,df2,df3,df4], axis = 1)

#%% Save to a dataframe

df_with_vis_descriptors.to_csv('Data_with_descriptors.csv', index = False)
    
    
        
        
        
        
        
        
    
        


        
        
        

        
    
    
    
