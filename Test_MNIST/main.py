import numpy as np
import pandas as pd
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = load_data()

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train / 255.
X_test = X_test / 255.

# plt.figure(figsize=(10, 2))
# for i in range(1, 11):
#     plt.subplot(1, 10, i)
#     plt.axis('off')
#     plt.imshow(X_train[i-1], cmap='gray_r')
#     plt.title(y_train[i-1], color='black', fontsize=16)
# plt.show()

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

results = model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test, y_test, verbose=2)

metrics = pd.DataFrame(results.history)

model.predict(X_test)

y_pred=model.predict(X_test)
y_classes=np.argmax(y_pred,axis=1)

pred = pd.concat([pd.DataFrame(y_test, columns=['y_test']), pd.DataFrame(y_classes, columns=['y_pred'])], axis=1)
print(pred.head(10))

misclassified = pred[pred['y_test'] != pred['y_pred']]
print(misclassified.index[:10])

plt.figure(figsize=(10, 2))
for i, j in zip(range(1, 11), misclassified.index[:10]):
    plt.subplot(1, 10, i)
    plt.axis('off')
    plt.imshow(X_test[j], cmap='gray_r')
    plt.title('y_test: ' + str(y_test[j]) + '\n' + 'y_pred: ' + str(y_classes[j]), color='black', fontsize=12)
plt.show()