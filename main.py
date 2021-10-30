
"""
Image Classification Project
"""

from data_handler import data_handler as dh

import pandas as pd

#%% Initial parameters

train_filePath = r"C:\Users\Pichau\Desktop\TensorFlow\ClassificationProject\Data\test"
superclass_index = 14 # 'people' superclass = 14

#%%

train_complete_dict = dh.importData(train_filePath)

#%% 

train_dict = dh.get_superclass(train_complete_dict, superclass_index)

#%%


