import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import timeit
from keras.models import load_model
import datetime
import PIL
from PIL import Image
import cv2
import numpy
    
def predictalpha(img,model):
##    model =load_model('alpha.h5')
##    model.get_weights()
##    model.optimizer


##    cv2.imwrite('as.jpg',img)
    image = Image.fromarray(img)
    
    img = image.resize((28,28), Image.ANTIALIAS)
    x=(np.asarray(img))
    x=np.reshape(x,(1,28,28,1))
    prediction = model.predict(x)


    thresholded = (prediction>0.5)*1
    
    
    digit=np.where(thresholded == 1)[1][0]


    classes = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
        25: "Z"
    
    }
    return classes[digit]


def predictnum(img):
    model =load_model('number.h5')
    model.get_weights()
    model.optimizer


##    cv2.imwrite('as.jpg',img)
    image = Image.fromarray(img)
##    image=img
##    print(type(image))
    img = image.resize((28,28), Image.ANTIALIAS)
    x=(np.asarray(img))
    x=np.reshape(x,(1,28,28,1))
    prediction = model.predict(x)


    thresholded = (prediction>0.5)*1
    
    
    digit=np.where(thresholded == 1)[1][0]


##    print(digit)
    return digit

def predictalphanum(img):
    model =load_model('alphanum.h5')
##    model.summary()
    model.get_weights()
    model.optimizer
##    model.summary()

    image = Image.fromarray(img)
    
    img = image.resize((32,32), Image.ANTIALIAS)
    x=(np.asarray(img))
    x=np.reshape(x,(1,32,32,1))
    prediction = model.predict(x)

    
    
##    image = Image.open(img_path,mode='r')
##    image = image.resize((32,32), Image.ANTIALIAS)
##    x = 
##    x=np.reshape(x,(1,28,28,1)) 
##    prediction = model.predict(x)
##    print(prediction)

    classes = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "A",
        11: "B",
        12: "C",
        13: "D",
        14: "E",
        15: "F",
        16: "G",
        17: "H",
        18: "I",
        19: "J",
        20: "K",
        21: "L",
        22: "M",
        23: "N",
        24: "O",
        25: "P",
        26: "Q",
        27: "R",
        28: "S",
        29: "T",
        30: "U",
        31: "V",
        32: "W",
        33: "X",
        34: "Y",
        35: "Z",
        36: "a",
        37: "b",
        38: "d",
        39: "e",
        40: "f",
        41: "g",
        42: "h",
        43: "n",
        44: "q",
        45: "r",
        46: "t"
    }
    thresholded = (prediction>0.5)*1
    return (classes[np.where(thresholded == 1)[1][0]])

if __name__ == "__main__":
    im = cv2.imread('1.png')    
    print(predictalpha(im))
##    print(predict('5.png'))
