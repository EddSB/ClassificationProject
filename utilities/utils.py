# -*- coding: utf-8 -*-
"""
Utiliy functions that do not fit in other categories
"""

import numpy as np


def rgb_to_gray_scale(rgb_img):
    """Transforms an images format from rgb to gray scale"""   
    
    rgb_img = rgb_img.reshape(3, 32, 32)
    rgb_img = rgb_img.transpose(1, 2, 0)
    
    b = [0.2989, 0.5870, 0.1140]
    
    gray_img = np.dot(rgb_img, b)
    
    gray_img = gray_img.reshape(1024)
    
    return gray_img