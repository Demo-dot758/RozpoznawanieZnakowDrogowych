import cv2
import numpy as np
from tensorflow.keras.models import load_model


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

image_dir = "images.jpg"
model = load_model('my_model_03.epoch41.hdf5')

image_list = []
image = cv2.imread(image_dir)

scale_percent = 250
width = int(image.shape[1] * scale_percent / 100) 
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
layers = resized.copy()

cv2.imshow('img',layers)
cv2.waitKey(0)
for i in range(4):
    layers = cv2.pyrDown(layers)
    image_list.append(layers)

winH, winW = 32,32

count = 1
for layer in image_list:
    count+=1
    for(x,y,window) in sliding_window(layer,stepSize=int(16/count),windowSize=(winH,winW)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        t_img = layer[y:y+winH,x:x+winW]
        test_img = np.expand_dims(t_img,axis =0) 
        prediction = model.predict(test_img, verbose = 0)
        clone = layer.copy()
        cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0, 255, 0), 2)
        predictions_per_location = np.argmax(prediction, 1)
        predictions_probability = np.max(prediction,axis=1)
        cv2.waitKey(0)
        if predictions_probability > 0.9:
            print("{}: {}".format(predictions_per_location, predictions_probability))
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.putText(clone, str(prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.destroyAllWindows()