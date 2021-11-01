# -*- coding: utf-8 -*-
"""
Module created to facilitate the visualization of data
"""
import numpy as np

import matplotlib.pyplot as plt
import configs.constants as CONST


def display_image(image, label = None):
    """Displays an image from RGB image array"""
    if (label != None):
        print("Image Class: ", CONST.CLASS_NAMES[label])
    plt.figure()
    plt.imshow(image)
    plt.show()
    
    
def compare_result(index, images, true_labels, predictions):
    """
    Displays the image RGB image and prints strings
    to compare the real label with the predicted label
    """
    predicted_label = np.argmax(predictions[index])
    print("Image class: ",
          CONST.CLASS_NAMES[np.int(true_labels[index])],
          ", predicted: ",
          CONST.CLASS_NAMES[predicted_label])
    display_image(images[index])
    
    
def display_gray_image(image):
    """Displays an image from grayscale image array"""
    image = image.reshape(32,32)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()
    