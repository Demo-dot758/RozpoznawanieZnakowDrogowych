import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping



generateadditional = ImageDataGenerator(rotation_range=10, zoom_range= 0.15, width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1, brightness_range=[1.0,1.2],horizontal_flip=False, vertical_flip=False, fill_mode="nearest",validation_split=0.2)

train_generator = generateadditional.flow_from_directory(
    directory='training2/',
    target_size=(64, 64),
    batch_size=32,
    subset='training',
    save_to_dir='newdata/')

validation_generator = generateadditional.flow_from_directory(
    directory='training2/',
    target_size=(64, 64),
    batch_size = 8,
    subset='validation')

epoch = 100
filepath = 'detection_newest_64.epoch{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

earlystop = EarlyStopping(monitor="val_loss",
                          patience = 5,
                          verbose = 1,
                          mode = 'min')
callbacks = [checkpoint, earlystop]

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(5,5), activation='relu', input_shape=(64,64,3)))
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))    
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=64, epochs=epoch, validation_data=validation_generator, callbacks=callbacks)

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()