# -*- coding: utf-8 -*-
"""
Module containing the different applicable TensorFlow models
"""

from tensorflow import keras


class Keras_sequential:
    
    model = keras.Sequential()
    
    def build(self):
        """Builds the Keras model"""
        self.model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(3072)),
            keras.layers.Dense(1000, activation=('relu')),
            keras.layers.Dense(100, activation=('relu')),
            keras.layers.Dense(5, activation=('softmax'))
        ])
        

    def compile_model(self):
        """Compiles the model"""
        self.model.compile(
            optimizer='adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
            )
        
        
    def train(self, _train_images, _train_labels, _epochs):
        """Trains the model"""
        self.model.fit(_train_images, _train_labels, epochs = _epochs)
        
    def test(self, _test_images, _test_labels):
        loss, acc = self.model.evaluate(_test_images, 
                                        _test_labels,
                                        verbose = 1)
        return (loss, acc)