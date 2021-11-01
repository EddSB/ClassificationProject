# -*- coding: utf-8 -*-
"""
Module containing the different applicable TensorFlow models
"""

from tensorflow import keras
from tensorflow.keras import layers
from abc import ABC, abstractmethod

class Model(ABC):
    
    @abstractmethod
    def build(self):
        pass
    
    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
        
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class Keras_sequential(Model):
    
    model = keras.Sequential()
    
    def build(self, output_layers=5):
        """Builds the Keras model"""
        self.model = keras.Sequential([
            #keras.layers.InputLayer(input_shape=(3072)),
            keras.layers.Flatten(),
            keras.layers.Dense(3072, activation=('relu')),
            keras.layers.Dense(1000, activation=('relu')),
            keras.layers.Dense(100, activation=('relu')),
            keras.layers.Dense(output_layers, activation=('softmax'))
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
    
    def evaluate(self, images, labels, verbose=2):
        """Evaluates the models accuracy"""
        loss, acc = self.model.evaluate(images, labels, verbose=2)
        return loss, acc
    
class Keras_convolutional(Model):
    
    model = keras.Sequential()
    
    def build(self, output_layers=5, filter_count1=32, 
              filter_count2=64, filter_count3=64):
        """Builds the Keras model with initial convolutional layers"""    
        self.model.add(layers.Conv2D(filter_count1, (2, 2),activation='relu', #(3,3)) 
                                     input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(filter_count2, (3, 3), activation='relu')) #(3,3)
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(filter_count3, (3, 3), activation='relu'))        
        self.model.add(layers.Flatten())
        #self.model.add(layers.Dense(300, activation='relu'))
        self.model.add(layers.Dense(200, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu')) #64
        self.model.add(layers.Dense(output_layers))
        
        
    def compile_model(self):
        """Compiles the model"""
        self.model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        
    def train(self, train_images, train_labels, epochs):
        """Trains the model"""
        # train_images = train_images.reshape(len(train_images), 3072)
        history = self.model.fit(train_images, train_labels, epochs = epochs)
        return history
        
    def test(self, _test_images, _test_labels):
        """Tests the model"""
        loss, acc = self.model.evaluate(_test_images, 
                                        _test_labels,
                                        verbose = 1)
        return (loss, acc)
    
        
    def predict(self, images):
        """Performs predictions on a set of images"""
        predictions = self.model.predict(images)
        return predictions
    
    def evaluate(self, images, labels, verbose=2):
        """Evaluates the models accuracy"""
        loss, acc = self.model.evaluate(images, labels, verbose=2)
        return loss, acc
        