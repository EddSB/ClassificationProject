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
        labels = _simplify_labels(labels)
    return (images, labels)


def import_cifar100_people():
    """
    Imports the data directly from the CIFAR-100 dataset
    available from the TensorFlow datasets, and returns only
    the 'people' superclass
    """
    
    ((train_images, train_labels), 
     (test_images, test_labels)) = datasets.cifar100.load_data() 
    ((_, train_coarse_labels),
     (_, test_coarse_labels)) = datasets.cifar100.load_data(label_mode='coarse')
    train_images, train_labels = _get_superclass_from_array(
        train_images, 
        train_labels,
        train_coarse_labels, 
        CONST.SUPERCLASS_INDEX
        )
    test_images, test_labels = _get_superclass_from_array(
        test_images, 
        test_labels,
        test_coarse_labels,
        CONST.SUPERCLASS_INDEX
        )
    train_labels = _simplify_labels(train_labels)
    test_labels = _simplify_labels(test_labels)
    return (train_images, test_images, train_labels, test_labels)


def import_cifar10():
    """
    Imports the data directly from the CIFAR-10 dataset
    available from the TensorFlow datasets, and returns only
    the 'people' superclass
    """
    
    ((train_images, train_labels), 
     (test_images, test_labels)) = datasets.cifar10.load_data() 

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


def _get_superclass_from_array(images, fine_labels, 
                               coarse_labels, superclass_index):
    """
    Recieves data in the form of an array and separates the
    desired superclass according to index
    """
    selected_images = [] 
    selected_labels = []   
    for i in range(len(images)):
        if (coarse_labels[i] == superclass_index):
            selected_images.append(images[i])
            selected_labels.append(fine_labels[i])
    selected_images = np.array(selected_images)
    selected_labels = np.array(selected_labels)
    return (selected_images, selected_labels)


def _normalize_pixel_values(image_list):
    """Normalizes pixel values from 0-255 to 0-1"""
    result = []
    for img in image_list:
        norm_img = img/255
        result.append(norm_img)
    return result


def _simplify_labels(labels):
    """
    Simplifies label values from arbitrarily distributed values to 
    sequential values starting at 0 
    """
    unique_labels = np.unique(labels)
    new_labels = np.zeros(len(labels))
    for i in range(len(labels)):
        for j in range(len(unique_labels)):
            if labels[i] == unique_labels[j]:
                new_labels[i] = j
    return new_labels
    

