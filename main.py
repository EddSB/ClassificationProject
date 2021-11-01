# -*- coding: utf-8 -*-
"""
Image Classification Project
"""

# import pandas as pd
# import numpy as np
# from tensorflow import keras

import data_handler.handler as dh
# import configs.constants as CONST
import models.tf_models as models
import utilities.visuals as vis
# import utilities.utils as utils

#%% Importing Data

#(train_images, train_labels) = dh.load_data(CONST.TRAIN_FILEPATH)
#(test_images, test_labels) = dh.load_data(CONST.TEST_FILEPATH)

#%% TESTING NEW DATA LOADING

(train_images, test_images, train_labels, test_labels) = dh.import_cifar100_people()

#%% Building Model
# I will start using a keras model

model = models.Keras_convolutional()
model.build()

#%% Compiling the Model

model.compile_model()

#%% Training the Model

history = model.train(train_images, train_labels, 6)

#%% Testing the model

print("\n Test Run:")
loss, acc = model.test(test_images, test_labels)
print("loss = ", loss, ", acc = ", acc)

#%% Predictiong one image

predictions = model.predict(test_images)

#%% Evaluating model

loss, acc = model.evaluate(test_images, test_labels)
print("\n Model accuracy: ", acc, "\n")

#%% Showing predictions

observed_index = 286
vis.compare_result(observed_index, test_images, test_labels, predictions)













