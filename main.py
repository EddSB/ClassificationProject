# -*- coding: utf-8 -*-
"""
Image Classification Project
"""

import numpy as np
import models.tf_models as models

import data_handler.handler as dh
import data_handler.augment as aug
import utilities.visuals as vis
import configs.variables as var

#%% 

if (var.dataset_choice == 1):
    (train_images, test_images, 
     train_labels, test_labels) = dh.import_cifar100_people()
elif (var.dataset_choice == 2):
    (train_images, test_images, 
     train_labels, test_labels) = dh.import_cifar10()


#%% 

train_images, train_labels = aug.augment(train_images, 
                                         train_labels, 
                                         var.augment_count)

#%%

model = models.Keras_convolutional()
classes_count = len(np.unique(test_labels))
model.build(classes_count, var.filter1, var.filter2, var.filter3)

#%%

model.compile_model()

#%%

history = model.train(train_images, train_labels, var.epochs)

#%%

print("\n Test Run:")
loss, acc = model.test(test_images, test_labels)
print("loss = ", loss, ", acc = ", acc)

#%% Predictiong one image

predictions = model.predict(test_images)

#%%

observed_index = 256
vis.compare_result(observed_index, test_images, test_labels, predictions)















