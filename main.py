
"""
Image Classification Project
"""

from data_loader import loader

import pandas as pd

#%% Initial parameters

train_filePath = r"C:\Users\Pichau\Desktop\TensorFlow\ClassificationProject\Data\test"
superclass_index = 14 # 'people' superclass = 14

#%%

train_complete_dict = loader.importData(train_filePath)

#%% 

train_dict = loader.get_superclass(train_complete_dict, superclass_index)

#%%



