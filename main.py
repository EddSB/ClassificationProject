# -*- coding: utf-8 -*-
"""
Image Classification Project
"""

# import pandas as pd
import numpy as np
# from tensorflow import keras

import data_handler.handler as dh
import data_handler.augment as aug
# import configs.constants as CONST
import models.tf_models as models
import utilities.visuals as vis
# import utilities.utils as utils

#%% 

# Input Variables
dataset_choice = 1       # 1 = CIFAR-10_People, 2 = CIFAR-100
filter1 = 32
filter2 = 64
filter3 = 64
epochs = 10
augment_count = 4

#%% 

if (dataset_choice == 1):
    (train_images, test_images, 
     train_labels, test_labels) = dh.import_cifar100_people()
elif (dataset_choice == 2):
    (train_images, test_images, 
     train_labels, test_labels) = dh.import_cifar10()


#%% 

train_images, train_labels = aug.augment(train_images, 
                                         train_labels, 
                                         augment_count)

#%%

model = models.Keras_convolutional()
classes_count = len(np.unique(test_labels))
model.build(classes_count, filter1, filter2, filter3)

#%%

model.compile_model()

#%%

history = model.train(train_images, train_labels, epochs)

#%%

print("\n Test Run:")
loss, acc = model.test(test_images, test_labels)
print("loss = ", loss, ", acc = ", acc)

#%% Predictiong one image

predictions = model.predict(test_images)

#%%

observed_index = 256
vis.compare_result(observed_index, test_images, test_labels, predictions)















