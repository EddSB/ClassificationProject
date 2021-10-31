# -*- coding: utf-8 -*-
"""
Module responsible for importing performing some initial processing of the 
Data
"""

import _pickle as pk
import numpy as np
import configs.constants as CONST

def import_data(filePath):
    with open(filePath, 'rb') as fo:
        data_dict = pk.load(fo, encoding='bytes') 
        data_dict = _get_superclass(data_dict, CONST.SUPERCLASS_INDEX)
        images = data_dict['data']
        images = _normalize_pixel_values(images)
        images = np.array(images)
        labels = data_dict['fine_labels']
        labels = _simplify_labels(labels, CONST.FINE_LABEL_NUMBERS)
    return (images, labels)

def _get_superclass(data_dict, superclass_index):
    result = {'data': [], 'fine_labels':[]}
    for i in range(len(data_dict[b'coarse_labels'])): #
        if (data_dict[b'coarse_labels'][i] == superclass_index):
            result['fine_labels'].append(data_dict[b'fine_labels'][i])
            result['data'].append(data_dict[b'data'][i].transpose())
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
    

