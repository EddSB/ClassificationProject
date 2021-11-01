# -*- coding: utf-8 -*-
"""
Module containing the different applicable TensorFlow models
"""

import tensorflow.keras as keras


class Keras_sequential:
    
    model = keras.Sequential()
    
    def build(self):
        """Builds the Keras model"""
        self.model = keras.Sequential([
            #keras.layers.InputLayer(input_shape=(3072)),
            keras.layers.Flatten(),
            keras.layers.Dense(3072, activation=('relu')),
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
        
        
    def train(self, train_images, train_labels, epochs):
        """Trains the model"""
        # train_images = train_images.reshape(len(train_images), 3072)
        self.model.fit(train_images, train_labels, epochs = epochs)
       
        
    def test(self, _test_images, _test_labels):
        """Tests the model"""
        loss, acc = self.model.evaluate(_test_images, 
                                        _test_labels,
                                        verbose = 1)
        return (loss, acc)
    
        
    def predict(self, images):
        """Evaluates the model's accuracy """
        predictions = self.model.predict(images)
        return predictions