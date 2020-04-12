# Number-Plate-Recognition

This is a project on Number Plate Recognition.

Steps involved in the Process :

1) Once a car is detected in the video.
2)The image of the car is cropped out.
3)The image of the car is further enhanced.
4)The Number Plate is detected in that image and it is extracted.
5)Using Character Level Segmentation,individual characters are extracted from the cropped number plate.
6)On the extracted characters,we further preprocess them and then use our CNN models to identify the characters
7)Then the extracted registration number is stored in a MYSQL database along with the timestamp.



The CNN models have been trained on the Kaggle and MNIST datasets for 200 epochs each on the Nvidia GPU.

