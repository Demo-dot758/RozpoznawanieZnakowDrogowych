import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

epoch = 100
filepath = 'my_model_03.epoch{epoch:02d}.hdf5'
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
data = []
labels = []
classes = [1,2,3,4,5,6,7,8,9]

num = 0

num2 = 0
for i in classes:
    path = os.path.abspath(os.getcwd()) + '/training'
    path = os.path.join(path,str(i))
    images = os.listdir(path)
    for a in images:
        try:
            if(i==4 and num % 2 == 0):
                image = Image.open(path + '/' + a)
                image = image.resize((32,32))
                image = np.array(image)
                data.append(image)
                labels.append(i)
                num+=1
                num2+=1
            elif(i == 4):
                num+=1
                continue
            else:
                image = Image.open(path + '/' + a)
                image = image.resize((32,32))
                image = np.array(image)
                data.append(image)
                labels.append(i)
        except:
            print("Error")    

data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)


X_train, _, y_t1, _ = train_test_split(data,labels,test_size=0.0000001, random_state=35, shuffle = True)

y_label = to_categorical(y_t1, 10)
print(X_train.shape,y_t1.shape)

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(5,5), activation='relu', input_shape=data.shape[1:]))
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))    
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

generateadditional = ImageDataGenerator(rotation_range=10, zoom_range= 0.20, width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1, brightness_range=[1.0,1.2],horizontal_flip=False, vertical_flip=False, fill_mode="nearest",validation_split=0.2)
train_generator = generateadditional.flow(
    x=X_train,
    y=y_label,
    batch_size=32,
    subset='training')

validation_generator = generateadditional.flow(
    x=X_train,
    y=y_label,
    batch_size = 32,
    subset='validation')

history = model.fit(train_generator, steps_per_epoch=64, epochs=epoch, validation_data=validation_generator, callbacks=callbacks)

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(num2)