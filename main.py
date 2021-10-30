
"""
Image Classification Project
"""

import pandas as pd
import numpy as np

from data_handler import handler as dh
from utilities import visuals
from configs import constants as const


#%%

train_dict = dh.importData(const.TRAIN_FILEPATH)
train_dict = dh.get_superclass(train_dict, const.SUPERCLASS_INDEX)
train_images = train_dict['data']
train_images = dh.normalize_pixel_values(train_images)
train_labels = train_dict['fine_labels']
train_labels = dh.simplify_labels(train_labels, const.FINE_LABEL_NUMBERS)


#%% Visualizando imagens

img_num = 21
visuals.display_image(train_images[img_num])

