# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:55:45 2021

@author: Pichau
"""

import _pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("Imported")

def importData(filePath):
    with open(filePath, 'rb') as fo:
        dict = pk.load(fo, encoding='bytes')
    return dict

def Test():
    print("This module works!")