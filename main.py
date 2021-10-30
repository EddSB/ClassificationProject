
"""
Image Classification Project
"""

# import pandas as pd
import numpy as np
# from tensorflow import keras

from data_handler import handler as dh
from utilities import visuals
from configs import constants as const
from models import tf_models 


#%%

train_dict = dh.importData(const.TRAIN_FILEPATH)
train_dict = dh.get_superclass(train_dict, const.SUPERCLASS_INDEX)
train_images = train_dict['data']
train_images = dh.normalize_pixel_values(train_images)
train_images = np.array(train_images)
train_labels = train_dict['fine_labels']
train_labels = dh.simplify_labels(train_labels, const.FINE_LABEL_NUMBERS)


#%% Image visualization

img_num = 22
visuals.display_image(train_images[img_num])

#%% Building Model
# I will start using a keras model

model = tf_models.Keras_sequential()
model.build


#%% Compiling the Model

model.model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#%% Training the Model

model.model.fit(train_images, train_labels, epochs = 30)
