import math
import cv2
import numpy as np
from predict import predictalpha,predictnum

def charSegmentation(img, kernelSize=23, sigma=11, theta=7, minArea=0,maxArea = 2000):
	"""Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
	
	Args:
		img: grayscale uint8 image of the text-line to be segmented.
		kernelSize: size of filter kernel, must be an odd integer.
		sigma: standard deviation of Gaussian function used for filter kernel.
		theta: approximated width/height ratio of words, filter function is distorted by this factor.
		minArea: ignore word candidates smaller than specified area.
		
	Returns:
		List of tuples. Each tuple contains the bounding box and the image of the segmented word.
	"""

	# apply filter kernel
	kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##	cv2.imshow('asgh',imgThres)
	imgThres = 255 - imgThres
##	cv2.imshow('as',imgThres)

	# find connected components. OpenCV: return type differs between OpenCV2 and 3
	if cv2.__version__.startswith('3.'):
		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# append components to result
	res = []
	for c in components:
		# skip small word candidates
		if cv2.contourArea(c) < minArea or cv2.contourArea(c)>maxArea:
			continue
		# append bounding box and image of word to result list
		currBox = cv2.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = img[y:y+h, x:x+w]
		res.append((currBox, currImg))

	# return list of words, sorted by x-coordinate
	return sorted(res, key=lambda entry:entry[0][0])


def prepareImg(img, height):
	"""convert given image to grayscale image (if needed) and resize to desired height"""
	assert img.ndim in (2, 3)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h = img.shape[0]
	factor = height / h
	return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel



def charsegcall(img):
	"""reads images from data/ and outputs the word-segmentation to out/"""

	# read image, prepare it by resizing it to fixed height and converting it to grayscale
##	img = prepareImg(cv2.imread(img), 50)

	img = prepareImg(img,50)
##	img=np.pad(img,(10,10),'constant',constant_values=(255))
##	cv2.imwrite('pas.jpg',img)
	# execute segmentation with given parameters
	# -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
	# -theta: approximated width/height ratio of words, filter function is distorted by this factor
	# - minArea: ignore word candidates smaller than specified area
	ret,img = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
	res = charSegmentation(img, kernelSize=5, sigma=35, theta=7, minArea=50,maxArea=300)
	

	listchar =[]
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		wordImg=np.pad(wordImg,(10,10),'constant',constant_values=(255))
		(x, y, w, h) = wordBox
##		listchar.append(wordImg)
##		print("len:",len(listchar))
		ret,wordImg = cv2.threshold(wordImg,110,255,cv2.THRESH_BINARY)

		x = wordImg
		ret,thresh1 = cv2.threshold(x,110,255,cv2.THRESH_BINARY)
		img=thresh1
		blur = cv2.blur(img,(2,2))
		img=blur
		img00=np.uint8(np.log1p(img))
		wordImg = cv2.normalize(img00, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
		_, img3 = cv2.threshold(wordImg, 250, 255, cv2.THRESH_BINARY)
##                cv2.imshow('log',img3)
                
		listchar.append(img3)
##		cv2.imwrite('%d.png'%(1+ j+100), img3) # save word
		
##		cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
		

        
	return listchar

if __name__ == '__main__':
        img =cv2.imread('plate1.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img,60, 255, cv2.THRESH_BINARY)[1]
        listchar=charsegcall(img)
        print(len(listchar))
        cv2.imwrite('img.jpg',img)
        for i in range(len(listchar)):
                cv2.imwrite(str(i)+'s.jpg',listchar[i])
##        print(predict(listc))
        word =' '
        word += str(predictalpha(cv2.threshold(listchar[0], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictalpha(cv2.threshold(listchar[1], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictnum(cv2.threshold(listchar[2], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictnum(cv2.threshold(listchar[3], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictalpha(cv2.threshold(listchar[4], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictalpha(cv2.threshold(listchar[4], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictnum(cv2.threshold(listchar[5], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictnum(cv2.threshold(listchar[6], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictnum(cv2.threshold(listchar[7], 150, 255, cv2.THRESH_BINARY_INV)[1]))
        word += str(predictnum(cv2.threshold(listchar[8], 150, 255, cv2.THRESH_BINARY_INV)[1]))
##        word+=predictalpha(listc[i])
        print(word)
##        print(len(listc))
##        print(listc[0].shape)
