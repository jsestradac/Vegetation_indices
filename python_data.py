# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:59:28 2025

@author: Robotics
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import ast 

def prepare_data (data, y_value='FMC_d'):
    #Ingresa el pd.series y lo regresa como array para entrenar el modelo o 
    #para evaluar
    
    
    x_data = data['spectral']
    y_data = data[y_value]

    
    x_data = x_data.to_numpy()
    np_data = [ast.literal_eval(s) for s in x_data]
    np_data = np.array(np_data)
    x = np_data.reshape(np_data.shape[0], np_data.shape[1])
    y = y_data.to_numpy()/100.0
    return x, y  

def compute_vi(spectral):
    """
    spectral: numpy array of shape (n_samples, n_wavelengths)
    returns: dict with vegetation indices as numpy arrays (n_samples,)
    """
    spectral = np.asarray(spectral)
    d = 350
    zeros = np.zeros((spectral.shape[0], d))
    R = np.concatenate([zeros, spectral], axis=1)

    vi = {
        # Ratio indices
        "SRPI1": R[:, 430] / R[:, 680],
        "SRPI2": R[:, 750] / R[:, 556],
        "SRPI3": R[:, 750] / R[:, 680],
        "BGI1": R[:, 400] / R[:, 550],
        "BGI2": R[:, 420] / R[:, 554],
        "BGI3": R[:, 450] / R[:, 550],
        "BRI1": R[:, 400] / R[:, 690],
        "BRI2": R[:, 450] / R[:, 690],
        "RGI1": R[:, 690] / R[:, 550],
        "RGI2": R[:, 695] / R[:, 445],
        "RBI":  R[:, 695] / R[:, 445],
        "SPADI": R[:, 650] / R[:, 940],
        "LIC1": R[:, 690] / R[:, 440],
        "FRI1": R[:, 690] / R[:, 600],
        "FRI2": R[:, 740] / R[:, 800],
        "CTR1": R[:, 695] / R[:, 420],
        "CTR2": R[:, 695] / R[:, 760],
        "CTR3": R[:, 750] / R[:, 695],
        "VOG1": R[:, 740] / R[:, 720],
        "GM1": R[:, 750] / R[:, 550],
        "GM2": R[:, 750] / R[:, 700],
        "RVI1": R[:, 750] / R[:, 705],
        "RVI2": R[:, 800] / R[:, 550],
        "RVI3": R[:, 800] / R[:, 635],
        "RVI4": R[:, 800] / R[:, 680],
        "PARSA": R[:, 750] / R[:, 710],
        "PARSC": R[:, 760] / R[:, 500],
        "PARSCD": R[:, 760] / R[:, 515],
        "PARSAD": R[:, 780] / R[:, 720],
        "PSSRA": R[:, 800] / R[:, 650],
        "PSSRB": R[:, 800] / R[:, 650],
        "PSSRC": R[:, 800] / R[:, 470],
        "DD": R[:, 850] / R[:, 710],

        # Modified simple ratio
        "CLRD": R[:, 750]/R[:, 710] - 1,
        "CL2D": R[:, 760]/R[:, 710] - 1,
        "CL3RE": R[:, 800]/R[:, 710] - 1,
        "CLG": R[:, 800]/R[:, 550] - 1,
        "CRI1": 1/R[:, 510] - 1/R[:, 550],
        "CRI2": 1/R[:, 510] - 1/R[:, 700],
        "ANTG": R[:, 780]*(1/R[:, 550] - 1/R[:, 700]),
        "ANT1": 1/R[:, 550] - 1/R[:, 700],
        "ANT2": R[:, 800]*(1/R[:, 550] - 1/R[:, 700]),
        "ANT3": R[:, 776]*(1/R[:, 530] - 1/R[:, 673]),
        "PARSA2": R[:, 675]/(R[:, 650]*R[:, 700]),
        "PARSB": R[:, 675]/(R[:, 640]*R[:, 705]),
        "CHLA": R[:, 776]*(1/R[:, 673] - 1),
        "CHLB": R[:, 776]*(1/R[:, 625] - 1/R[:, 673]),
        "PSRI": (R[:, 680]-R[:, 500])/R[:, 750],
        "RVSI": 0.5*(R[:, 722]+R[:, 763]) - R[:, 733],

        # Normalized difference
        "PQ": (R[:, 415]-R[:, 435])/(R[:, 415]+R[:, 435]),
        "NPQD": (R[:, 482]-R[:, 350])/(R[:, 482]+R[:, 350]),
        "PRI": (R[:, 531]-R[:, 570])/(R[:, 531]+R[:, 570]),
        "PRID": (R[:, 531]-R[:, 580])/(R[:, 531]+R[:, 580]),
        "NPCI": (R[:, 680]-R[:, 430])/(R[:, 680]+R[:, 480]),
        "NDVI1": (R[:, 750]-R[:, 680])/(R[:, 750]+R[:, 780]),
        "NDVI2": (R[:, 750]-R[:, 705])/(R[:, 750]+R[:, 705]),
        "NDVI3D": (R[:, 780]-R[:, 715])/(R[:, 780]+R[:, 715]),
        "NDVI4": (R[:, 800]-R[:, 670])/(R[:, 800]+R[:, 670]),
        "NDVI5": (R[:, 800]-R[:, 550])/(R[:, 800]+R[:, 550]),
        "NDVI6": (R[:, 800]-R[:, 700])/(R[:, 800]+R[:, 700]),
        "NDVI7": (R[:, 850]-R[:, 680])/(R[:, 850]+R[:, 680]),
        "PSNDA": (R[:, 800]-R[:, 680])/(R[:, 800]+R[:, 680]),
        "PSNDB": (R[:, 800]-R[:, 635])/(R[:, 800]+R[:, 635]),
        "PSNDC": (R[:, 800]-R[:, 460])/(R[:, 800]+R[:, 460]),
        "PSNDCD": (R[:, 800]-R[:, 482])/(R[:, 800]+R[:, 482]),
        "LIC2": (R[:, 790]-R[:, 680])/(R[:, 790]+R[:, 680]),

        # Modified normalized difference
        "VOG2": (R[:, 734]-R[:, 747])/(R[:, 715]+R[:, 720]),
        "VOG3": (R[:, 734]-R[:, 747])/(R[:, 715]+R[:, 726]),
        "MSR1": (R[:, 750]-R[:, 445])/(R[:, 705]-R[:, 445]),
        "MSR2": (R[:, 780]-R[:, 710])/(R[:, 780]-R[:, 680]),
        "MSR3": (R[:, 850]-R[:, 710])/(R[:, 850]-R[:, 680]),
        "SIPI": (R[:, 800]-R[:, 445])/(R[:, 800]-R[:, 680]),
        "MDATT1": (R[:, 703]-R[:, 732])/(R[:, 703]-R[:, 722]),
        "MDATT2": (R[:, 705]-R[:, 732])/(R[:, 705]-R[:, 722]),
        "MDATT3": (R[:, 710]-R[:, 727])/(R[:, 710]-R[:, 734]),
        "MDATT4": (R[:, 712]-R[:, 744])/(R[:, 712]-R[:, 720]),
        "MDATT5": (R[:, 719]-R[:, 732])/(R[:, 719]-R[:, 743]),
        "MDATT6": (R[:, 719]-R[:, 732])/(R[:, 719]-R[:, 726]),
        "MDATT7": (R[:, 719]-R[:, 742])/(R[:, 719]-R[:, 732]),
        "MDATT8": (R[:, 719]-R[:, 747])/(R[:, 719]-R[:, 721]),
        "MDATT9": (R[:, 719]-R[:, 761])/(R[:, 719]-R[:, 493]),
        "MDATT10": (R[:, 721]-R[:, 744])/(R[:, 721]-R[:, 714]),
        "MDATT11": (R[:, 688]-R[:, 744])/(R[:, 688]-R[:, 736]),
        
        
        # Water related indices
        "WBI": R[:, 970] / R[:, 900],
        "MSI": R[:, 1600] / R[:, 820],
        "MSI1": R[:, 1650] / R[:, 1230],
        "MSI2": R[:, 1650] / R[:, 830],
        "TM57": R[:, 1650] / R[:, 2220],
        "WI": R[:, 900] / R[:, 970],
        "FWBI": R[:, 900] / np.minimum(R[:, 930], R[:, 980]),
        "LWI": R[:, 1300] / R[:, 1450],
        "SWRI": R[:, 860] / R[:, 1240],
        "SWRI1": R[:, 1350] / R[:, 870],
        "SRWI2": R[:, 880] / R[:, 1265],
        "NDWI": (R[:, 850] - R[:, 1650]) / (R[:, 850] + R[:, 1650]),
        "NDWI1": (R[:, 860] - R[:, 1260]) / (R[:, 870] + R[:, 1260]),
        "NDWI2": (R[:, 870] - R[:, 1260]) / (R[:, 870] + R[:, 1260]),
        "SIWSI": (R[:, 1640] - R[:, 858]) / (R[:, 1640] + R[:, 858]),
        "DDI": 2*R[:, 1530] - R[:, 1005] - R[:, 2055],
        "MAX": R[:, 2500],
    
    }

    return vi



#%% load data avocado
olive_spectral = loadmat('olive/Spectral_olive.mat')
olive_chlorophyll = loadmat('olive/Chlorophyll_olive.mat')
olive_nitrogen = loadmat('olive/Nitrogen_olive.mat')
olive_FMC = loadmat('olive/FMC_olive.mat')

species_column_olive = ['olive' for x in range(111*5)]
fresh_column_olive = ['Fresh' for x in range(111)]
stage1_column_olive = ['Stage1' for x in range(111)]
stage2_column_olive = ['Stage2' for x in range(111)]
stage3_column_olive = ['Stage3' for x in range(111)]
dry_column_olive = ['Dry' for x in range(111)]

olive_stages = [*fresh_column_olive, 
                  *stage1_column_olive,
                  *stage2_column_olive,
                  *stage3_column_olive,
                  *dry_column_olive]

olive_fresh = olive_spectral['fresh_leaves_olive']
olive_stage1 = olive_spectral['stage1_leaves_olive']
olive_stage2 = olive_spectral['stage2_leaves_olive']
olive_stage3 = olive_spectral['stage3_leaves_olive']
olive_dry = olive_spectral['dry_leaves_olive']

spectral_olive = np.concatenate((olive_fresh, olive_stage1, olive_stage2,
                          olive_stage3, olive_dry))

chlorophyll_olive = olive_chlorophyll['chlorophyll_olive']

chlorophyll_olive = np.concatenate((chlorophyll_olive[:,0],
                              chlorophyll_olive[:,1],
                              chlorophyll_olive[:,2],
                              chlorophyll_olive[:,3],
                              chlorophyll_olive[:,4]), axis = 0)

FMC_f_olive = olive_FMC['FMC_f_olive']
FMC_f_olive = np.concatenate((FMC_f_olive[:,0],
                        FMC_f_olive[:,1],
                        FMC_f_olive[:,2],
                        FMC_f_olive[:,3],
                        FMC_f_olive[:,4]), axis = 0)

FMC_d_olive = olive_FMC['FMC_d_olive']
FMC_d_olive = np.concatenate((FMC_d_olive[:,0],
                        FMC_d_olive[:,1],
                        FMC_d_olive[:,2],
                        FMC_d_olive[:,3],
                        FMC_d_olive[:,4]), axis = 0)


#%%

avocado_spectral = loadmat('Avocado/Spectral_avocado.mat')
avocado_chlorophyll = loadmat('Avocado/Chlorophyll_avocado.mat')
avocado_nitrogen = loadmat('Avocado/Nitrogen_avocado.mat')
avocado_FMC = loadmat('Avocado/FMC_avocado.mat')

species_column_avocado = ['Avocado' for x in range(104*5)]
fresh_column_avocado = ['Fresh' for x in range(104)]
stage1_column_avocado = ['Stage1' for x in range(104)]
stage2_column_avocado = ['Stage2' for x in range(104)]
stage3_column_avocado = ['Stage3' for x in range(104)]
dry_column_avocado = ['Dry' for x in range(104)]

avocado_stages = [*fresh_column_avocado, 
                  *stage1_column_avocado,
                  *stage2_column_avocado,
                  *stage3_column_avocado,
                  *dry_column_avocado]

avocado_fresh = avocado_spectral['fresh_leaves_avocado']
avocado_stage1 = avocado_spectral['stage1_leaves_avocado']
avocado_stage2 = avocado_spectral['stage2_leaves_avocado']
avocado_stage3 = avocado_spectral['stage3_leaves_avocado']
avocado_dry = avocado_spectral['dry_leaves_avocado']

spectral_avocado = np.concatenate((avocado_fresh, avocado_stage1, avocado_stage2,
                          avocado_stage3, avocado_dry))

chlorophyll_avocado = avocado_chlorophyll['chlorophyll_avocado']

chlorophyll_avocado = np.concatenate((chlorophyll_avocado[:,0],
                              chlorophyll_avocado[:,1],
                              chlorophyll_avocado[:,2],
                              chlorophyll_avocado[:,3],
                              chlorophyll_avocado[:,4]), axis = 0)

FMC_f_avocado = avocado_FMC['FMC_f_avocado']
FMC_f_avocado = np.concatenate((FMC_f_avocado[:,0],
                        FMC_f_avocado[:,1],
                        FMC_f_avocado[:,2],
                        FMC_f_avocado[:,3],
                        FMC_f_avocado[:,4]), axis = 0)

FMC_d_avocado = avocado_FMC['FMC_d_avocado']
FMC_d_avocado = np.concatenate((FMC_d_avocado[:,0],
                        FMC_d_avocado[:,1],
                        FMC_d_avocado[:,2],
                        FMC_d_avocado[:,3],
                        FMC_d_avocado[:,4]), axis = 0)

#%%

vineyard_spectral = loadmat('vineyard/Spectral_vineyard.mat')
vineyard_chlorophyll = loadmat('vineyard/Chlorophyll_vineyard.mat')
vineyard_nitrogen = loadmat('vineyard/Nitrogen_vineyard.mat')
vineyard_FMC = loadmat('vineyard/FMC_vineyard.mat')

species_column_vineyard = ['vineyard' for x in range(104*5)]
fresh_column_vineyard = ['Fresh' for x in range(104)]
stage1_column_vineyard = ['Stage1' for x in range(104)]
stage2_column_vineyard = ['Stage2' for x in range(104)]
stage3_column_vineyard = ['Stage3' for x in range(104)]
dry_column_vineyard = ['Dry' for x in range(104)]

vineyard_stages = [*fresh_column_vineyard, 
                  *stage1_column_vineyard,
                  *stage2_column_vineyard,
                  *stage3_column_vineyard,
                  *dry_column_vineyard]

vineyard_fresh = vineyard_spectral['fresh_leaves_vineyard']
vineyard_stage1 = vineyard_spectral['stage1_leaves_vineyard']
vineyard_stage2 = vineyard_spectral['stage2_leaves_vineyard']
vineyard_stage3 = vineyard_spectral['stage3_leaves_vineyard']
vineyard_dry = vineyard_spectral['dry_leaves_vineyard']

spectral_vineyard = np.concatenate((vineyard_fresh, vineyard_stage1, vineyard_stage2,
                          vineyard_stage3, vineyard_dry))

chlorophyll_vineyard = vineyard_chlorophyll['chlorophyll_vineyard']

chlorophyll_vineyard = np.concatenate((chlorophyll_vineyard[:,0],
                              chlorophyll_vineyard[:,1],
                              chlorophyll_vineyard[:,2],
                              chlorophyll_vineyard[:,3],
                              chlorophyll_vineyard[:,4]), axis = 0)

FMC_f_vineyard = vineyard_FMC['FMC_f_vineyard']
FMC_f_vineyard = np.concatenate((FMC_f_vineyard[:,0],
                        FMC_f_vineyard[:,1],
                        FMC_f_vineyard[:,2],
                        FMC_f_vineyard[:,3],
                        FMC_f_vineyard[:,4]), axis = 0)

FMC_d_vineyard = vineyard_FMC['FMC_d_vineyard']
FMC_d_vineyard = np.concatenate((FMC_d_vineyard[:,0],
                        FMC_d_vineyard[:,1],
                        FMC_d_vineyard[:,2],
                        FMC_d_vineyard[:,3],
                        FMC_d_vineyard[:,4]), axis = 0)

#%%

stages = [*fresh_column_avocado,
          *stage1_column_avocado,
          *stage2_column_avocado,
          *stage3_column_avocado,
          *dry_column_avocado,
          *fresh_column_olive,
          *stage1_column_olive,
          *stage2_column_olive,
          *stage3_column_olive,
          *dry_column_olive,
          *fresh_column_vineyard,
          *stage1_column_vineyard,
          *stage2_column_vineyard,
          *stage3_column_vineyard,
          *dry_column_vineyard]

spectral = np.concatenate((spectral_avocado, spectral_olive, spectral_vineyard), axis = 0)
species = [*species_column_avocado,
           *species_column_olive,
           *species_column_vineyard]

FMC_f = np.concatenate((FMC_f_avocado, FMC_f_olive, FMC_f_vineyard), axis = 0)
FMC_d = np.concatenate((FMC_d_avocado, FMC_d_olive, FMC_d_vineyard), axis = 0)

chlorophyll = np.concatenate((chlorophyll_avocado, chlorophyll_olive, chlorophyll_vineyard))

vis = compute_vi(spectral)
spectral_list = spectral.tolist()
dataframe = {
                 'Species': species,
                 'FMC_f': FMC_f,
                 'FMC_d': FMC_d,
                 'chlorophyll': chlorophyll,
                 'spectral': spectral_list
                
     }

total_dataframe = dataframe | vis

dataframe = pd.DataFrame(total_dataframe)

# dataframe.to_csv('output.csv',index=False)

#%%

# df = pd.read_csv('output.csv', index_col= False)
# spectral = df['spectral']
# line = spectral[0]



# x, y = prepare_data(df)





