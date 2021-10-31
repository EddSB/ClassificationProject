# -*- coding: utf-8 -*-
"""
Image Classification Project
"""

# import pandas as pd
# import numpy as np
# from tensorflow import keras

import data_handler.handler as dh
import configs.constants as CONST
import models.tf_models as models
from utilities import visuals

#%%


(train_images, train_labels) = dh.import_data(CONST.TRAIN_FILEPATH)
(test_images, test_labels) = dh.import_data(CONST.TEST_FILEPATH)

#%% Image visualization

img_num = 69
visuals.display_image(test_images[img_num])

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
