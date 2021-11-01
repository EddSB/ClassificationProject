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

#%% Variables

dataset_choice = 1       # 1 = CIFAR-10_People, 2 = CIFAR-100

filter1 = 32
filter2 = 64
filter3 = 64

epochs = 10


#%% TESTING NEW DATA LOADING

if (dataset_choice == 1):
    (train_images, test_images, 
     train_labels, test_labels) = dh.import_cifar100_people()
elif (dataset_choice == 2):
    (train_images, test_images, 
     train_labels, test_labels) = dh.import_cifar10()


#%% TESTING DATA AUGMENTATION

train_images, train_labels = aug.augment(train_images, train_labels, 5)

#%% Visualization

i = 1454
vis.display_image(train_images[i], np.int(train_labels[i]))


#%% Building Model
# I will start using a keras model

model = models.Keras_convolutional()
output_neuron_count = len(np.unique(test_labels))
model.build(output_neuron_count, filter1, filter2, filter3)

#%% Compiling the Model

model.compile_model()

#%% Training the Model

history = model.train(train_images, train_labels, epochs)

#%% Testing the model

print("\n Test Run:")
loss, acc = model.test(test_images, test_labels)
print("loss = ", loss, ", acc = ", acc)

#%% Evaluating model

loss, acc = model.evaluate(test_images, test_labels)
print("\n Model accuracy: ", acc, "\n")

#%% Predictiong one image

predictions = model.predict(test_images)

#%% Showing predictions

# observed_index = 256
# vis.compare_result(observed_index, test_images, test_labels, predictions)















