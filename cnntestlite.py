
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import time


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
        string = "Droga z pierszeństwem"
        return string
    if(string == 4):
        string = "Stop"
        return string
    if(string == 5):
        string = "Zakaz wjazdu"
        return string
    if(string == 6):
        string = "Nakaz skrętu w prawo"
        return string
    if(string == 7):
        string = "Nakaz skrętu w lewo"
        return string
    if(string == 8):
        string = "Nakaz jazdy prosto"
        return string
    if(string == 9):
        string = "Inny"
        return string

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

y_test = pd.read_csv('Test22.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
# predictions=[]
correct_images = []
incorrect_images = []

# get the start time
st = time.time()

for img,label in zip(imgs,labels):
    image = Image.open(img).convert('RGB')
    image = image.resize((64,64))
    input_data = np.expand_dims(image,axis=0)
    input_data = np.array(input_data,dtype=np.float32)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    y_classes=np.argmax(output,axis=1)
    y_pred_conf = np.max(output,axis=1)
    predicted_label = labelstonames(y_classes)
    # predictions.append(predicted_label)

    if y_classes == label:
        correct_images.append((image, label))
    else:
        incorrect_images.append((image, label))

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st

print(elapsed_time)

whole = len(incorrect_images) + len(correct_images)


    
accuracy = len(correct_images) / whole * 100

print("Accuracy", accuracy , "%")
