import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from keras import layers
from keras import activations
from keras import backend as K

def labelstonames(string):
    if(string == 0):
        string = "Ograniczenie do 30 km/h"
        return string
    if(string == 1):
        string = "Ograniczenie do 50 km/h"
        return string
    if(string == 2):
        string = "Ograniczenie do 120 km/h"
        return string
    if(string == 3):
        string = "Droga z pierszenstwem"
        return string
    if(string == 4):
        string = "Stop"
        return string
    if(string == 5):
        string = "Zakaz wjazdu"
        return string
    if(string == 6):
        string = "Nakaz skretu w prawo"
        return string
    if(string == 7):
        string = "Nakaz skretu w lewo"
        return string
    if(string == 8):
        string = "Nakaz jazdy prosto"
        return string
    if(string == 9):
        string = "Inny"
        return string


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

image_dir = "IMG20221106173325.jpg"

model = load_model('model/pro_crop_git12\detection_new.epoch31.h5')

image_list = []
image = cv2.imread(image_dir)

scale_percent = 35

width = int(image.shape[1] * scale_percent / 100) 
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

for i in range(1):
    resized = cv2.pyrDown(resized)
    image_list.append(resized)

# cv2.imshow('img',image_list[0])
# cv2.waitKey(0)
winH, winW = 64,64

count = 0                                                                               

pred_list = []
for layer in image_list:
    for(x,y,window) in sliding_window(layer,stepSize=int(8/count),windowSize=(winH,winW)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        t_img = layer[y:y+winH,x:x+winW]
        test_img = np.expand_dims(t_img,axis = 0) 
        prediction = model.predict(test_img, verbose = 0)
        clone = layer.copy()
        predictions_per_location = np.argmax(prediction, 1)
        predictions_probability = np.max(prediction,axis = 1)
        if predictions_probability > 0.9 and predictions_per_location != 9:
            # func = K.function([model.input], [model.layers[-2].output])
            # outputs = func(test_img)
            # print(outputs)
            pred_list.append((predictions_probability, predictions_per_location, (x,y), (winW,winH)))
            # print("{}: {}".format(labelstonames(predictions_per_location), predictions_probability))
            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            # cv2.putText(clone, str(labelstonames(predictions_per_location)), (x-20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.imshow("Window", clone)
            # cv2.waitKey(0)
    count+=1


highest_prediction = max(pred_list, key=lambda x: x[0])

prediction_prob = highest_prediction[0]
predicted_sign = highest_prediction[1]
x,y = highest_prediction[2]
winW,winH = highest_prediction[3]

cv2.rectangle(image_list[0]), (x, y), (x + winW, y + winH), (0, 255, 0), 2)
cv2.putText(image_list[0]), str(labelstonames(predicted_sign)), (x-20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imshow("Window", image_list[0])
cv2.waitKey(0)

cv2.destroyAllWindows()