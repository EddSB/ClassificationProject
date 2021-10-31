# -*- coding: utf-8 -*-
"""
Utiliy functions that do not fit in other categories
"""

import numpy as np


def rgb_list_to_gray_scale(rgb_images):
    gray_images = []
    for img in rgb_images:
        gray_images.append(_rgb_img_to_gray_scale(img))
    gray_images = np.array(gray_images)
    return gray_images


def _rgb_img_to_gray_scale(rgb_img):
    """Transforms an images format from rgb to gray scale"""   
    rgb_img = rgb_img.reshape(3, 32, 32)
    rgb_img = rgb_img.transpose(1, 2, 0)
    weights = [0.2989, 0.5870, 0.1140]
    gray_img = np.dot(rgb_img, weights) 
    gray_img = gray_img.reshape(1024)
    return gray_img

