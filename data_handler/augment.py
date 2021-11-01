# -*- coding: utf-8 -*-
"""
Module Responsible  Data Augmentation, generating transforms of existing 
images.
"""

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# import matplotlib.pyplot as plt

def augment(images, labels, num_transforms):
    """
    Augments data by creating x transforms of each image, and 
    concatenating this images onto the original array
    """
    size = len(images)
    
    for i in range(size):
        # Create a data generator object that transforms images
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
        img = images[i]
        img = img.reshape((1,) + img.shape)
        
        i = 0
        transformed_img = []
        transformed_labels = []
        
        for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
            transformed_img.append(image.img_to_array(batch[0]))
            transformed_labels.append(labels[20])
            i += 1
            if i >= num_transforms:       # Repeat 'num_transforms' times
                break

        images = np.vstack((images, np.array(transformed_img)))
        labels = np.concatenate((labels, transformed_labels), axis = 0)
    
    return images, labels
    