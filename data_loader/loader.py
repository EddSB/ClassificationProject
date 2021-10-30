# -*- coding: utf-8 -*-
"""
Module responsible for importing performing some initial processing of the 
Data
"""

import _pickle as pk

def importData(filePath):
    with open(filePath, 'rb') as fo:
        dict = pk.load(fo, encoding='bytes')
    return dict

def get_superclass(data_dict, superclass_index):    
    _result = {'filenames': [], 'data': [], 'fine_labels':[]}
    for i in range(len(data_dict[b'coarse_labels'])): #
        if data_dict[b'coarse_labels'][i] == superclass_index :
            _result['filenames'].append(data_dict[b'filenames'][i])
            _result['fine_labels'].append(data_dict[b'fine_labels'][i])
            _result['data'].append(data_dict[b'data'][i].transpose())
    return _result