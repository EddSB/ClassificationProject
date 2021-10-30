# -*- coding: utf-8 -*-
"""
Module responsible for importing performing some initial processing of the 
Data
"""

import _pickle as pk
import numpy as np

def importData(filePath):
    with open(filePath, 'rb') as fo:
        dict = pk.load(fo, encoding='bytes')
    return dict


def get_superclass(data_dict, superclass_index):    
    result = {'data': [], 'fine_labels':[]}
    for i in range(len(data_dict[b'coarse_labels'])): #
        if (data_dict[b'coarse_labels'][i] == superclass_index):
            result['fine_labels'].append(data_dict[b'fine_labels'][i])
            result['data'].append(data_dict[b'data'][i].transpose())
    return result


def normalize_pixel_values(image_list):
    result = []
    for img in image_list:
        norm_img = img/255
        result.append(norm_img)
    return result


def simplify_labels(train_labels, original_labels):
    new_labels = np.zeros(len(train_labels))
    for i in range(len(train_labels)):
        for j in range(len(original_labels)):
            if train_labels[i] == original_labels[j]:
                new_labels[i] = j
    return new_labels
    