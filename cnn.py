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

data = []
labels = []
classes = [1,2,3,4,5,6,7,8,9]

for i in classes:
    path = os.path.abspath(os.getcwd()) + '/train'
    path = os.path.join(path,str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '/' + a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error")    

data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

X_t1, X_t2, y_t1, y_t2 = train_test_split(data,labels,test_size=0.33, random_state=9)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)

y_t1 = to_categorical(y_t1, 10)
y_t2 = to_categorical(y_t2, 10)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_t1.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epoch = 30

generateadditional = ImageDataGenerator(rotation_range=10, zoom_range= 0.10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, horizontal_flip=False, vertical_flip=False, fill_mode="nearest")
history = model.fit(generateadditional.flow(X_t1, y_t1, batch_size=32), epochs=epoch, validation_data=(X_t2, y_t2))
model.save("road_signs0.h5")

score = model.evaluate(X_t2, y_t2, verbose=9)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())
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