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
import utilities.visuals as vis

#%%


(train_images, train_labels) = dh.import_data(CONST.TRAIN_FILEPATH)
(test_images, test_labels) = dh.import_data(CONST.TEST_FILEPATH)


#%%
img_num = 1
#%% Image visualization

img_num = 2
vis.display_image(train_images[img_num], np.int(train_labels[img_num]))

#%% Building Model
# I will start using a keras model

model = models.Keras_sequential()
model.build()


#%% Compiling the Model

model.compile_model()

#%% Training the Model

model.train(train_images, train_labels, 10)

#%% Testing the model

print("\n Test Run:")

loss, acc = model.test(test_images, test_labels)

print("loss = ", loss, ", acc = ", acc)
# model.test(test)

#%% Checking one image

predictions = model.predict(train_images)

#%% Showing predictions

observed_index = 184
predict_index = np.argmax(predictions[observed_index])

i = np.argmax( predictions[observed_index])

vis.comparative(train_images[observed_index],
                train_labels[observed_index],
                predict_index)














