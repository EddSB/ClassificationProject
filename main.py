# -*- coding: utf-8 -*-
"""
Image Classification Project
"""

# import pandas as pd
import numpy as np
# from tensorflow import keras

import data_handler.handler as dh
import configs.constants as CONST
import models.tf_models as models
# from utilities import visuals

#%%

train_dict = dh.importData(CONST.TRAIN_FILEPATH)
train_dict = dh.get_superclass(train_dict, CONST.SUPERCLASS_INDEX)
train_images = train_dict['data']
train_images = dh.normalize_pixel_values(train_images)
train_images = np.array(train_images)
train_labels = train_dict['fine_labels']
train_labels = dh.simplify_labels(train_labels, CONST.FINE_LABEL_NUMBERS)


#%% Image visualization

# img_num = 22
# visuals.display_image(train_images[img_num])

#%% Building Model
# I will start using a keras model

model = models.Keras_sequential()
model.build()


#%% Compiling the Model

model.compile_model()

#%% Training the Model

model.train(train_images, train_labels, 10)

#%% Testing the model

# model.test(test)