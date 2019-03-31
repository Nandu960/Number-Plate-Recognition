
import cv2 
from imutils import paths
import numpy as np
import imutils
from bin import smooth1
import datetime
from prjct import connect,add
import time

from WordSegmentation import charsegcall
from predict import predictalpha,predictnum,predictalphanum
from number_plate2 import preprocess,cleanAndRead,extract_contours
from keras.models import load_model

now = datetime.datetime.now()

cap = cv2.VideoCapture('8r1.mp4') 
car_cascade = cv2.CascadeClassifier('cars.xml') 

connect()

model1 =load_model('alpha.h5')
model1.get_weights()
model1.optimizer
model1.summary()

model2 =load_model('number.h5')
model2.get_weights()
model2.optimizer
model2.summary()

while True:
    
    ret, frames = cap.read()
    if (ret==False):
        break;

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)


    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    
    for (x,y,w,h) in cars: 
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
        crop_img = frames[y:y+h, x:x+w]

    img = crop_img
    threshold_img = preprocess(img)
    contours= extract_contours(threshold_img)
    ol =cleanAndRead(img,contours)
##    cv2.imshow('video',frames)

    img = cv2.threshold(ol, 150, 255, cv2.THRESH_BINARY)[1]
    word=''
    if ol is not None:
        listchar=charsegcall(ol)
##        print(len(listchar))
        if (len(listchar)>=10):
            print('Plate Detected')
            
            cv2.imwrite('as.jpg',cv2.threshold(listchar[0], 150, 255, cv2.THRESH_BINARY_INV)[1])
            word += str(predictalpha(cv2.threshold(listchar[0], 150, 255, cv2.THRESH_BINARY_INV)[1],model1))
            word += str(predictalpha(cv2.threshold(listchar[1], 150, 255, cv2.THRESH_BINARY_INV)[1],model1))
            word += str(predictnum(cv2.threshold(listchar[2], 150, 255, cv2.THRESH_BINARY_INV)[1]))
            word += str(predictnum(cv2.threshold(listchar[3], 150, 255, cv2.THRESH_BINARY_INV)[1]))
            word += str(predictalpha(cv2.threshold(listchar[4], 150, 255, cv2.THRESH_BINARY_INV)[1],model1))
            word += str(predictalpha(cv2.threshold(listchar[5], 150, 255, cv2.THRESH_BINARY_INV)[1],model1))
            word += str(predictnum(cv2.threshold(listchar[6], 150, 255, cv2.THRESH_BINARY_INV)[1]))
            word += str(predictnum(cv2.threshold(listchar[7], 150, 255, cv2.THRESH_BINARY_INV)[1]))
            word += str(predictnum(cv2.threshold(listchar[8], 150, 255, cv2.THRESH_BINARY_INV)[1]))
            word += str(predictnum(cv2.threshold(listchar[9], 150, 255, cv2.THRESH_BINARY_INV)[1]))
            
######            for i in range(len(listchar)):
########                cv2.imwrite('as'+str(i)+'.jpg',listchar[i])
######                if i>9:
######                    img = cv2.threshold(listchar[i], 150, 255, cv2.THRESH_BINARY_INV)[1]
######                    word+=str(predictnum(img))
            now = datetime.datetime.now()
            add(word,str(now.strftime("%H:%M:%S")),str(now.date()))
            print(word)
        else:
            print("Plate Not Detected")

##    cv2.imshow('video2', frames) 
      
    # Wait for Esc key to stop 
    if cv2.waitKey(33) == 27: 
        break
        









  

cv2.destroyAllWindows() 
