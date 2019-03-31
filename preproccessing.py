import cv2
from matplotlib import pyplot as plt
import numpy as np

x = cv2.imread('1.png')
cv2.imshow('original',x)
ret,thresh1 = cv2.threshold(x,110,255,cv2.THRESH_BINARY)
cv2.imshow('as',thresh1)
cv2.imwrite('binimg.jpg',thresh1)

img=thresh1

blur = cv2.blur(img,(2,2))
##blur = cv2.GaussianBlur(img,(3,3),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
img=blur


img00=np.uint8(np.log1p(img))
cv2.imshow('log1',img00)
n_image = cv2.normalize(img00, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
cv2.imshow('norm',n_image)
##_, img3 = cv2.threshold(n_image, 250, 255, cv2.THRESH_BINARY)
##cv2.imshow('log',img3)
##cv2.imwrite('blurlog.jpg',img3)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##img=img3

### Create our shapening kernel, it must equal to one eventually
##kernel_sharpening = np.array([[1,1,1], 
##                              [1, -8,1],
##                              [1,1,1]])
### applying the sharpening kernel to the input image & displaying it.
##sharpened = cv2.filter2D(img, -1, kernel_sharpening)
##cv2.imshow('Image Sharpening', sharpened)
##cv2.imwrite('imgshar.jpg',sharpened)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
