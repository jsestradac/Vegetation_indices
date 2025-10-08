# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 13:40:51 2025

@author: Robotics
"""

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import ast
import cv2 as cv


def prepare_data (data):
    #Ingresa el pd.series y lo regresa como array para entrenar el modelo o 
    #para evaluar
    #los arrays se guardan como 
    
    
    x_data = data['Reflectance']
    y_data = data['FMC']

    
    x_data = x_data.to_numpy()
    np_data = [ast.literal_eval(s) for s in x_data]
    np_data = np.array(np_data)
    x = np_data.reshape(np_data.shape[0], np_data.shape[1])
    y = y_data.to_numpy()/100.0
    return x, y

def get_maximum_size(dir_imgs):
    
    max_x = 0
    max_y = 0
    max_x_path = ''
    max_y_path = ''    
    
    for path in dir_imgs:
        img = cv.imread(path, -1)
        x,y = img.shape
        
        if x > max_x:
            max_x = x
            max_x_path = path
            
            
        if y > max_y:
            max_y = y 
            max_y_path = path
            
    return max_x, max_y, max_x_path, max_y_path

#%%

df = pd.read_csv('Dataset_with_images.csv')

blue_images = df['blue']
green_images = df['green']
red_images = df['red']
nir_images = df['nir']
re_images = df['red_edge']

total_images = pd.concat([blue_images, green_images, 
                          red_images, nir_images, re_images], axis = 0)

total_images = total_images.tolist()

#max_x, max_y, max_x_path, max_y_path = get_maximum_size(total_images)

#%%

def add_padding (max_x, max_y, img):
    
    x, y = img.shape
    
    dif_x = max_x - x
    dif_y = max_y - y 
    
    if (dif_x % 2) == 0:
        pad_left = int(dif_x/2)
        pad_right = int(dif_x/2)
        
    else:
        pad_left = int(np.floor(dif_x/2))
        pad_right = int(np.floor(dif_x/2)+1)
        
    if (dif_y % 2) == 0:
        pad_down = int(dif_y/2)
        pad_up = int(dif_y/2)
        
    else:
        pad_down = int(np.floor(dif_y/2))
        pad_up = int(np.floor(dif_y/2)+1)
        
    padd_left = np.zeros([pad_left, y])
    padd_right = np.zeros([pad_right, y])
    
    img_padded_x = np.concatenate([padd_left, img, padd_right])
    
    padd_up = np.zeros([img_padded_x.shape[0], pad_up])
    padd_down = np.zeros([img_padded_x.shape[0], pad_down])
    
    img_padded = np.concatenate([padd_up, img_padded_x, padd_down], axis=1)
    
    return img_padded

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


#%%

def safe_divide(numerator, denominator):
    """Divide arrays safely: returns NaN when denominator is zero or invalid."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = 0 # replace inf and -inf with nan
    return result

def ndvi(b,g,r,nir,re):
    return safe_divide(nir - r, nir + r)

def gndvi(b,g,r,nir,re):
    return safe_divide(nir - g, nir + g)

def ndre(b,g,r,nir,re):
    return safe_divide(nir - re, nir + re)

def sipi(b,g,r,nir,re):
    return safe_divide(nir - b, nir + b)

def ngbdi(b,g,r,nir,re):
    return safe_divide(g - b, g + b)

def ngrdi(b,g,r,nir,re):
    return safe_divide(g - r, g + r)

def grdi(b,g,r,nir,re):
    return g - r

def nbgvi(b,g,r,nir,re):
    return safe_divide(b - g, b + g)

def negi(b,g,r,nir,re):
    return safe_divide(2*g - r - b, 2*g + r + b)

def mgrvi(b,g,r,nir,re):
    return safe_divide(g**2 - r**2, g**2 + r**2)

def mvari(b,g,r,nir,re):
    return safe_divide(g - b, g + r - b)

def rgbvi(b,g,r,nir,re):
    return safe_divide(g**2 - b*r, g**2 + b*r)

def tgi(b,g,r,nir,re):
    return g - 0.39*r - 0.61*b

def vari(b,g,r,nir,re):
    return safe_divide(g - r, g + r - b)

def grri(b,g,r,nir,re):
    return safe_divide(g, r)

def nri(b,g,r,nir,re):
    return safe_divide(r, r + g + b)    

def grvi(b,g,r,nir,re):
    return safe_divide(g - r, g + r - b)

def sr(b,g,r,nir,re):
    return safe_divide(nir, r)

def savi(b,g,r,nir,re):
    return safe_divide(1.5*(nir - r), nir + r + 0.5)

def cl(b,g,r,nir,re):
    return safe_divide(nir, re) - 1

def psri(b,g,r,nir,re):
    return safe_divide(r - g, re)

def m3cl(b,g,r,nir,re):
    return safe_divide(nir + r + re, nir - r + re)

def si(b,g,r,nir,re):
    return (r + g + b)/3

def msr(b,g,r,nir,re):
    return safe_divide(safe_divide(nir, r) - 1, safe_divide(nir, r) + 1)

def osavi(b,g,r,nir,re):
    return safe_divide(1.16*(nir - r), nir + r + 0.16)

def rvi(b,g,r,nir,re):
    return safe_divide(nir, r)

def rvi2(b,g,r,nir,re):
    return safe_divide(nir, g)

def tvi(b,g,r,nir,re):
    return 60*(nir - g) - 100*(g - r)

def evi(b,g,r,nir,re):
    return safe_divide(2.5*(nir - r), nir + 6*r - 7.5*b + 1)

def gi(b,g,r,nir,re):
    return safe_divide(g, r)

def tcari(b,g,r,nir,re):
    return 3*((re - r) - 0.2*(re - g)*safe_divide(re, r))

def srpi(b,g,r,nir,re):
    return safe_divide(b, r)

def npci(b,g,r,nir,re):
    return safe_divide(r - b, r + b)

def ndvigb(b,g,r,nir,re):
    return safe_divide(g - b, g + b)

def psri2(b,g,r,nir,re):
    return safe_divide(b - r, g)

def cive(b,g,r,nir,re):
    return 0.44*r - 0.81*g + 0.39*b + 18.79

def nirv(b,g,r,nir,re):
    return nir * ndvi(b,g,r,nir,re)

def dvi(b,g,r,nir,re):
    return nir - r

def msavi(b,g,r,nir,re):
    return safe_divide((2*nir + 1) - np.sqrt((2*nir + 1)**2 - 8*(nir - r)), 2)

def cari(b,g,r,nir,re):
    return re - r - 0.2*(re - g)

def remsr(b,g,r,nir,re):
    return safe_divide(safe_divide(nir, re) - 1, safe_divide(np.sqrt(nir), re) + 1)

def rendvi(b,g,r,nir,re):
    return safe_divide(nir - re, nir + re)

def lci(b,g,r,nir,re):
    return safe_divide(nir - re, nir + r)

#%%
from scipy import stats

def get_descriptors(img):
    #img = np.asanyarray(img)
    
    img = img/(2**16-1)
    nonzero_mask = img != 0
    count = nonzero_mask.sum()
    
    if count == 0:
        return 0,0,0
    
    total = img[nonzero_mask].sum()
    median = np.median(img[nonzero_mask])
    mode = stats.mode(img[nonzero_mask],keepdims=False).mode
    return total/count, median, mode



#%%

import os
from pathlib import Path
from tqdm import tqdm


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
    
    b = add_padding(max_x, max_y, b)
    g = add_padding(max_x, max_y, g)
    r = add_padding(max_x, max_y, r)
    nir = add_padding(max_x, max_y, nir)
    re = add_padding(max_x, max_y, re)
    
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
    
#%%

df2 = pd.DataFrame(average)
df3 = pd.DataFrame(median)
df4 = pd.DataFrame(mode)

df_with_vis_descriptors = pd.concat([df,df2,df3,df4], axis = 1)

#%%

df_with_vis_descriptors.to_csv('Data_with_descriptors.csv', index = False)
    
    
        
        
        
        
        
        
    
        


        
        
        

        
    
    
    
