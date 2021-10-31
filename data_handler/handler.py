# -*- coding: utf-8 -*-
"""
Module responsible for importing performing some initial processing of the 
Data
"""

from tensorflow.keras import datasets
import _pickle as pk
import numpy as np
import configs.constants as CONST

def load_data(filePath):
    """Loads the data available locally"""
    with open(filePath, 'rb') as fo:
        data_dict = pk.load(fo, encoding='bytes') 
        data_dict = _get_superclass_from_dict(data_dict, CONST.SUPERCLASS_INDEX)
        images = data_dict['data']
        images = _normalize_pixel_values(images)
        images = np.array(images)
        labels = data_dict['fine_labels']
        labels = _simplify_labels(labels, CONST.FINE_LABEL_NUMBERS)
    return (images, labels)

def import_cifar100_people():
    """
    Imports the data directly from the CIFAT-100 dataset
    available from the TensorFlow datasets, and returns only
    the 'people' superclass
    """
    ((train_images, train_labels), 
     (test_images, test_labels)) = datasets.cifar100.load_data()
    ((_, train_coarse_labels), 
     (_, test_coarse_labels)) = datasets.cifar100.load_data(
         label_mode = 'coarse')
    train_images = _get_superclass_from_array(train_images, 
                                              train_coarse_labels, 
                                              CONST.SUPERCLASS_INDEX)
    test_images = _get_superclass_from_array(test_images, 
                                             test_coarse_labels, 
                                             CONST.SUPERCLASS_INDEX)
    return (train_images, test_images, train_labels, test_labels)
        
    
def _get_superclass_from_dict(data_dict, superclass_index):
    """
    Recieves data in the form of a dictionary and separates the
    desired superclass according to index
    """
    result = {'data': [], 'fine_labels':[]}
    for i in range(len(data_dict[b'coarse_labels'])): #
        if (data_dict[b'coarse_labels'][i] == superclass_index):
            result['fine_labels'].append(data_dict[b'fine_labels'][i])
            result['data'].append(data_dict[b'data'][i].transpose())
    return result


def _get_superclass_from_array(images, coarse_labels, superclass_index):
    """
    Recieves data in the form of an array and separates the
    desired superclass according to index
    """
    result = []
    for i in range(len(images)): #
        if (coarse_labels[i] == superclass_index):
            result.append(images[i])
    result = np.array(result)        
    return result


def _normalize_pixel_values(image_list):
    result = []
    for img in image_list:
        norm_img = img/255
        result.append(norm_img)
    return result


def _simplify_labels(train_labels, original_labels):
    new_labels = np.zeros(len(train_labels))
    for i in range(len(train_labels)):
        for j in range(len(original_labels)):
            if train_labels[i] == original_labels[j]:
                new_labels[i] = j
    return new_labels
    

