
"""
Image Classification Project
"""

from data_handler import handler as dh
from utilities import visuals
from configs import constants as const

import pandas as pd
import numpy as np


#%%
# train_filePath = r"C:\Users\Pichau\Desktop\TensorFlow\ClassificationProject\Data\test"
# superclass_index = 14 # 'people' superclass = 14

#%%

train_complete_dict = dh.importData(const.TRAIN_FILEPATH)

#%% 

train_dict = dh.get_superclass(train_complete_dict, const.SUPERCLASS_INDEX)
train_images = train_dict['data']
train_labels = train_dict['fine_labels']


#%% Normalizing Values

dh.normalize_pixel_values(train_images)

#%% Visualizando imagens

img_num = 21
visuals.display_image(train_images[img_num])

