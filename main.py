
"""
Image Classification Project
"""

from data_handler import handler as dh
from utilities import visuals

import pandas as pd
import numpy as np


#%% Initial parameters

train_filePath = r"C:\Users\Pichau\Desktop\TensorFlow\ClassificationProject\Data\test"
superclass_index = 14 # 'people' superclass = 14

#%%

train_complete_dict = dh.importData(train_filePath)

#%% 

train_dict = dh.get_superclass(train_complete_dict, superclass_index)
train_images = train_dict['data']

#%% WORK IN PROGRESS

# Normalizing Values

dh.normalize_pixel_values(train_images)
print (dh)

#%% Visualizando imagens

img_num = 21
visuals.display_image(train_images[img_num])

