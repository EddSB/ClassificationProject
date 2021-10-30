# -*- coding: utf-8 -*-
"""
Module created to facilitate the visualization of data
"""

def display_image(image_array):
    import matplotlib.pyplot as plt
    plt.figure()
    image = image_array.reshape(3,32,32)
    image = image.transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()