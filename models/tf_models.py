# -*- coding: utf-8 -*-
"""
Module containing the different applicable TensorFlow models
"""

from tensorflow import keras


class Keras_sequential:
    
    model = keras.Sequential()
    
    def build(self):
        self.model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(3072)),
            keras.layers.Dense(1000, activation=('relu')),
            keras.layers.Dense(300, activation=('relu')),
            keras.layers.Dense(100, activation=('relu')),
            keras.layers.Dense(5, activation=('softmax'))
        ])
        

