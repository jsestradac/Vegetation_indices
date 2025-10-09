# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 18:31:15 2025

@author: Robotics
"""

import os 

avocado_folder = 'Avocado/Multispectral Images'
avocado_files = os.listdir(avocado_folder)
avocado_files = [os.path.join(avocado_folder, file) for file in avocado_files]

avocado_fresh_images = [file for file in avocado_files  if 'd0' in file]
avocado_stage1_images = [file for file in avocado_files  if 'd1' in file]
avocado_stage2_images = [file for file in avocado_files  if 'd2' in file]
avocado_stage3_images = [file for file in avocado_files  if 'd3' in file]
avocado_dry_images = [file for file in avocado_files  if 'd4' in file]

avocado_blue_fresh = [file for file in avocado_fresh_images if '_1' in file]
avocado_green_fresh = [file for file in avocado_fresh_images if '_2' in file]
avocado_red_fresh = [file for file in avocado_fresh_images if '_3' in file]
avocado_nir_fresh = [file for file in avocado_fresh_images if '_4' in file]
avocado_re_fresh = [file for file in avocado_fresh_images if '_5' in file]

avocado_blue_stage1 = [file for file in avocado_stage1_images if '_1' in file]
avocado_green_stage1 = [file for file in avocado_stage1_images if '_2' in file]
avocado_red_stage1 = [file for file in avocado_stage1_images if '_3' in file]
avocado_nir_stage1 = [file for file in avocado_stage1_images if '_4' in file]
avocado_re_stage1 = [file for file in avocado_stage1_images if '_5' in file]

avocado_blue_stage2 = [file for file in avocado_stage2_images if '_1' in file]
avocado_green_stage2 = [file for file in avocado_stage2_images if '_2' in file]
avocado_red_stage2 = [file for file in avocado_stage2_images if '_3' in file]
avocado_nir_stage2 = [file for file in avocado_stage2_images if '_4' in file]
avocado_re_stage2 = [file for file in avocado_stage2_images if '_5' in file]

avocado_blue_stage3 = [file for file in avocado_stage3_images if '_1' in file]
avocado_green_stage3 = [file for file in avocado_stage3_images if '_2' in file]
avocado_red_stage3 = [file for file in avocado_stage3_images if '_3' in file]
avocado_nir_stage3 = [file for file in avocado_stage3_images if '_4' in file]
avocado_re_stage3 = [file for file in avocado_stage3_images if '_5' in file]

avocado_blue_dry = [file for file in avocado_dry_images if '_1' in file]
avocado_green_dry = [file for file in avocado_dry_images if '_2' in file]
avocado_red_dry = [file for file in avocado_dry_images if '_3' in file]
avocado_nir_dry = [file for file in avocado_dry_images if '_4' in file]
avocado_re_dry = [file for file in avocado_dry_images if '_5' in file]


avocado_blue = [*avocado_blue_fresh,
                *avocado_blue_stage1,
                *avocado_blue_stage2,
                *avocado_blue_stage3,
                *avocado_blue_dry]

avocado_green = [*avocado_green_fresh,
                *avocado_green_stage1,
                *avocado_green_stage2,
                *avocado_green_stage3,
                *avocado_green_dry]

avocado_red = [*avocado_red_fresh,
                *avocado_red_stage1,
                *avocado_red_stage2,
                *avocado_red_stage3,
                *avocado_red_dry]

avocado_nir = [*avocado_nir_fresh,
                *avocado_nir_stage1,
                *avocado_nir_stage2,
                *avocado_nir_stage3,
                *avocado_nir_dry]

avocado_re = [*avocado_re_fresh,
                *avocado_re_stage1,
                *avocado_re_stage2,
                *avocado_re_stage3,
                *avocado_re_dry]

#%%
olive_folder = 'Olive/Multispectral Images'
olive_files = os.listdir(olive_folder)
olive_files = [os.path.join(olive_folder, file) for file in olive_files]

olive_fresh_images = [file for file in olive_files  if 'd0' in file]
olive_stage1_images = [file for file in olive_files  if 'd1' in file]
olive_stage2_images = [file for file in olive_files  if 'd2' in file]
olive_stage3_images = [file for file in olive_files  if 'd3' in file]
olive_dry_images = [file for file in olive_files  if 'd4' in file]

olive_blue_fresh = [file for file in olive_fresh_images if '_1' in file]
olive_green_fresh = [file for file in olive_fresh_images if '_2' in file]
olive_red_fresh = [file for file in olive_fresh_images if '_3' in file]
olive_nir_fresh = [file for file in olive_fresh_images if '_4' in file]
olive_re_fresh = [file for file in olive_fresh_images if '_5' in file]

olive_blue_stage1 = [file for file in olive_stage1_images if '_1' in file]
olive_green_stage1 = [file for file in olive_stage1_images if '_2' in file]
olive_red_stage1 = [file for file in olive_stage1_images if '_3' in file]
olive_nir_stage1 = [file for file in olive_stage1_images if '_4' in file]
olive_re_stage1 = [file for file in olive_stage1_images if '_5' in file]

olive_blue_stage2 = [file for file in olive_stage2_images if '_1' in file]
olive_green_stage2 = [file for file in olive_stage2_images if '_2' in file]
olive_red_stage2 = [file for file in olive_stage2_images if '_3' in file]
olive_nir_stage2 = [file for file in olive_stage2_images if '_4' in file]
olive_re_stage2 = [file for file in olive_stage2_images if '_5' in file]

olive_blue_stage3 = [file for file in olive_stage3_images if '_1' in file]
olive_green_stage3 = [file for file in olive_stage3_images if '_2' in file]
olive_red_stage3 = [file for file in olive_stage3_images if '_3' in file]
olive_nir_stage3 = [file for file in olive_stage3_images if '_4' in file]
olive_re_stage3 = [file for file in olive_stage3_images if '_5' in file]

olive_blue_dry = [file for file in olive_dry_images if '_1' in file]
olive_green_dry = [file for file in olive_dry_images if '_2' in file]
olive_red_dry = [file for file in olive_dry_images if '_3' in file]
olive_nir_dry = [file for file in olive_dry_images if '_4' in file]
olive_re_dry = [file for file in olive_dry_images if '_5' in file]


olive_blue = [*olive_blue_fresh,
                *olive_blue_stage1,
                *olive_blue_stage2,
                *olive_blue_stage3,
                *olive_blue_dry]

olive_green = [*olive_green_fresh,
                *olive_green_stage1,
                *olive_green_stage2,
                *olive_green_stage3,
                *olive_green_dry]

olive_red = [*olive_red_fresh,
                *olive_red_stage1,
                *olive_red_stage2,
                *olive_red_stage3,
                *olive_red_dry]

olive_nir = [*olive_nir_fresh,
                *olive_nir_stage1,
                *olive_nir_stage2,
                *olive_nir_stage3,
                *olive_nir_dry]

olive_re = [*olive_re_fresh,
                *olive_re_stage1,
                *olive_re_stage2,
                *olive_re_stage3,
                *olive_re_dry]
#%%

vineyard_folder = 'Vineyard/Multispectral Images'
vineyard_files = os.listdir(vineyard_folder)
vineyard_files = [os.path.join(vineyard_folder, file) for file in vineyard_files]

vineyard_fresh_images = [file for file in vineyard_files  if 'd0' in file]
vineyard_stage1_images = [file for file in vineyard_files  if 'd1' in file]
vineyard_stage2_images = [file for file in vineyard_files  if 'd2' in file]
vineyard_stage3_images = [file for file in vineyard_files  if 'd3' in file]
vineyard_dry_images = [file for file in vineyard_files  if 'd4' in file]

vineyard_blue_fresh = [file for file in vineyard_fresh_images if '_1' in file]
vineyard_green_fresh = [file for file in vineyard_fresh_images if '_2' in file]
vineyard_red_fresh = [file for file in vineyard_fresh_images if '_3' in file]
vineyard_nir_fresh = [file for file in vineyard_fresh_images if '_4' in file]
vineyard_re_fresh = [file for file in vineyard_fresh_images if '_5' in file]

vineyard_blue_stage1 = [file for file in vineyard_stage1_images if '_1' in file]
vineyard_green_stage1 = [file for file in vineyard_stage1_images if '_2' in file]
vineyard_red_stage1 = [file for file in vineyard_stage1_images if '_3' in file]
vineyard_nir_stage1 = [file for file in vineyard_stage1_images if '_4' in file]
vineyard_re_stage1 = [file for file in vineyard_stage1_images if '_5' in file]

vineyard_blue_stage2 = [file for file in vineyard_stage2_images if '_1' in file]
vineyard_green_stage2 = [file for file in vineyard_stage2_images if '_2' in file]
vineyard_red_stage2 = [file for file in vineyard_stage2_images if '_3' in file]
vineyard_nir_stage2 = [file for file in vineyard_stage2_images if '_4' in file]
vineyard_re_stage2 = [file for file in vineyard_stage2_images if '_5' in file]

vineyard_blue_stage3 = [file for file in vineyard_stage3_images if '_1' in file]
vineyard_green_stage3 = [file for file in vineyard_stage3_images if '_2' in file]
vineyard_red_stage3 = [file for file in vineyard_stage3_images if '_3' in file]
vineyard_nir_stage3 = [file for file in vineyard_stage3_images if '_4' in file]
vineyard_re_stage3 = [file for file in vineyard_stage3_images if '_5' in file]

vineyard_blue_dry = [file for file in vineyard_dry_images if '_1' in file]
vineyard_green_dry = [file for file in vineyard_dry_images if '_2' in file]
vineyard_red_dry = [file for file in vineyard_dry_images if '_3' in file]
vineyard_nir_dry = [file for file in vineyard_dry_images if '_4' in file]
vineyard_re_dry = [file for file in vineyard_dry_images if '_5' in file]


vineyard_blue = [*vineyard_blue_fresh,
                *vineyard_blue_stage1,
                *vineyard_blue_stage2,
                *vineyard_blue_stage3,
                *vineyard_blue_dry]

vineyard_green = [*vineyard_green_fresh,
                *vineyard_green_stage1,
                *vineyard_green_stage2,
                *vineyard_green_stage3,
                *vineyard_green_dry]

vineyard_red = [*vineyard_red_fresh,
                *vineyard_red_stage1,
                *vineyard_red_stage2,
                *vineyard_red_stage3,
                *vineyard_red_dry]

vineyard_nir = [*vineyard_nir_fresh,
                *vineyard_nir_stage1,
                *vineyard_nir_stage2,
                *vineyard_nir_stage3,
                *vineyard_nir_dry]

vineyard_re = [*vineyard_re_fresh,
                *vineyard_re_stage1,
                *vineyard_re_stage2,
                *vineyard_re_stage3,
                *vineyard_re_dry]
#%%

blue = [*avocado_blue,
        *olive_blue,
        *vineyard_blue]

green = [*avocado_green,
         *olive_green,
         *vineyard_green]

red = [*avocado_red,
       *olive_red,
       *vineyard_red]

nir = [*avocado_nir,
       *olive_nir,
       *vineyard_nir]

red_edge = [*avocado_re,
            *olive_re,
            *vineyard_re]


images = {'blue' : blue,
          'green' : green,
          'red' : red,
          'nir' : nir,
          'red_edge': red_edge 
    
    }
import pandas as pd

df_images = pd.DataFrame(images)

df = pd.read_csv('total_data.csv', index_col= False)

df_new_total = pd.concat([df, df_images], axis=1)

#%%
# df_new_total.to_csv('Dataset_with_images.csv')