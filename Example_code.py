# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:35:41 2025

@author: Robotics
"""

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import ast
import cv2 as cv
from scipy import stats
import os
from tqdm import tqdm

from utils import *


dataset = pd.read_csv('Data_with_descriptors.csv')

#spectral = get_spectral_data(dataset)

spectral_avocado = dataset[dataset['Species'] == 'Avocado']

spectral_avocado = get_spectral_data(spectral_avocado)

plt.plot(spectral_avocado[0])
#%%



green_images_list = dataset['green']
green_images_list = green_images_list.to_list()

img0 = cv.imread(green_images_list[0],-1) #it is important to load 
plt.imshow(img0,cmap='gray')
