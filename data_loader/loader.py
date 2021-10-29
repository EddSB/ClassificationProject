# -*- coding: utf-8 -*-
"""
Class responsible for import the Data
"""

import _pickle as pk

def importData(filePath):
    with open(filePath, 'rb') as fo:
        dict = pk.load(fo, encoding='bytes')
    return dict