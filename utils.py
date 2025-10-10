# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 14:04:29 2025

@author: Robotics
"""

import pandas as pd 
import numpy as np 
import ast
import cv2 as cv
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

def add_padding (img, max_x=731, max_y=888):
    
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

def get_spectral_data(data):
    
    
    x_data = data['spectral']
       
    x_data = x_data.to_list()
    np_data = [ast.literal_eval(s) for s in x_data]
    np_data = np.array(np_data)
    x = np_data.reshape(np_data.shape[0], np_data.shape[1])
    #y = y_data.to_numpy()/100.0
    return x


