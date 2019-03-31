import cv2
import numpy as np
from matplotlib import pyplot as plt
##from predict import predict

def binim(img,thresh):
##    img = cv2.imread(img_path,0)
    ret,thresh1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)
##    ret1,thresh1 = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
    
 
##    cv2.imwrite(img_path,thresh1)
    return thresh1


def smooth(img):
    img = cv2.medianBlur(img, 3)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(img,-1,kernel)

    img00=np.uint8(np.log1p(dst))

    n_image = cv2.normalize(img00, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    _, img3 = cv2.threshold(n_image, 254, 255, cv2.THRESH_BINARY)

    ret,thresh1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)

    return thresh1

def smooth1(img):
    img = cv2.medianBlur(img, 3)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(img,-1,kernel)

    img00=np.uint8(np.log1p(dst))

    n_image = cv2.normalize(img00, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    _, img3 = cv2.threshold(n_image, 254, 255, cv2.THRESH_BINARY)

    ret,thresh1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY)

    return thresh1



def binimchar(img,thresh):
##    img = cv2.imread('1.png')

    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(img,-1,kernel)
    
    img00=np.uint8(np.log1p(dst))
##cv2.imshow('log1',img00)
    n_image = cv2.normalize(img00, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
##cv2.imshow('norm',n_image)
    _, img3 = cv2.threshold(n_image, 252, 255, cv2.THRESH_BINARY)


    ret,thresh1 = cv2.threshold(img3,180,255,cv2.THRESH_BINARY_INV)
    return thresh1

def gauss(img):
    blur = cv2.blur(img,(2,2))
    ##blur = cv2.GaussianBlur(img,(3,3),0)

    img=blur


    img00=np.uint8(np.log1p(img))
    n_image = cv2.normalize(img00, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    _, img3 = cv2.threshold(n_image, 249, 255, cv2.THRESH_BINARY)

    ret,thresh1 = cv2.threshold(img3,180,255,cv2.THRESH_BINARY_INV)
    
##
### Create our shapening kernel, it must equal to one eventually
##    kernel_sharpening = np.array([[1,1,1], 
##                                  [1, -8,1],
##                                  [1,1,1]])
### applying the sharpening kernel to the input image & displaying it.
##    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    return thresh1
    
##cv2.imshow('log',img3)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

##if __name__ == "__main__":
####    binim('0.png',170)
##    cv2.imwrite('b1.png',binim('1.png',170))
##    cv2.imwrite('b2.png',binim('2.png',170))
##    binim('3.png',170)
####    binim('4.png',170)
####    binim('5.png')
####    binim('6.png')
##    print(predict(binim('1.png',170)))
##    print(predict(binim('2.png',170)))
##    print(predict(binim('3.png',170)))
    

    
