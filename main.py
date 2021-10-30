
"""
Image Classification Project
"""

from data_loader import loader

import pandas as pd

#%% Initial parameters

data_filePath = r"C:\Users\Pichau\Desktop\TensorFlow\ClassificationProject\Data\test"
superclass_index = 14 # 'people' superclass = 14

#%% 

data_dict = loader.importData(data_filePath)

#%% 

people_dict = loader.get_superclass(superclass_index)

#%%

