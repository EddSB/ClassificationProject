# -*- coding: utf-8 -*-
"""
Module created to facilitate the visualization of data
"""

import matplotlib.pyplot as plt
import configs.constants as CONST

def display_image(image, label):
    print("Image class: ", CONST.CLASS_NAMES[label])
    plt.figure()
    image = image.reshape(3,32,32)
    image = image.transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()
    
def comparative(image, true_label, predicted_label):
    print("Image class: ",
          CONST.CLASS_NAMES[true_label],
          ", predicted: ",
          CONST.CLASS_NAMES[predicted_label])
    plt.figure()
    image = image.reshape(3,32,32)
    image = image.transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()