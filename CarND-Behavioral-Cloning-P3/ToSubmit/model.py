import csv
import cv2
import numpy as np

# This is the list of directories containing input data
parent_dir_1 = '../data_0723_0206/'
parent_dir_2 = '../data/'

# This is added/subtracted from the steering_measurement while using left/right
# camera images
steering_correction_factor = 0.2

# cropping factors
crop_top = 70
crop_bottom = 20

# parameter to control augmenting data using multiple cameras
use_multiple_cameras = True

# code to augment data by flipping each image horizontally and negating the measurement
def augment_data(images, measurements):
	augmented_images = []
	augmented_measurements = []
	for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_images.append(cv2.flip(image,1))
		augmented_measurements.append(measurement)
		augmented_measurements.append(measurement*-1.0)
	return augmented_images, augmented_measurements

# Top level function to get training data for images and csv files in parent_dir    
def get_training_data(parent_dir):
	data_dir = parent_dir + 'IMG/'
	csv_file = parent_dir + 'driving_log.csv'
	images = []
	measurements = []
	with open(csv_file) as file:
		reader = csv.reader(file)
		for line in reader:
			if line[0] == 'center':
				continue
			fname = data_dir + line[0].split('/')[-1]
			image = cv2.imread(fname)
			image = image[...,::-1] #bgr to rgb (as simulator video is in rgb)
			images.append(image)
			measurements.append(float(line[3]))
			# using left/right camera images
			if use_multiple_cameras:
				# add left camera image
				fname = data_dir + line[1].split('/')[-1]
				image = cv2.imread(fname)
				image = image[...,::-1] #bgr to rgb
				images.append(image)
				measurements.append(float(line[3])+steering_correction_factor)
				# add right camera image
				fname = data_dir + line[2].split('/')[-1]
				image = cv2.imread(fname)
				image = image[...,::-1] #bgr to rgb
				images.append(image)
				measurements.append(float(line[3])-steering_correction_factor)
	# Data augmentation
	images, measurements = augment_data(images, measurements)
	X_train = np.array(images)
	y_train= np.array(measurements)
	return X_train, y_train

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Flatten, Dense, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Basic single layer neural network
def get_basic_model(input_shape=(160,320,3)):
	model = Sequential()
	model.add(Lambda(lambda x : (x/255.0)-0.5, input_shape=input_shape))
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(1))
	return model
	
# Lenet convolutional neural network
def get_lenet_model(input_shape=(160,320,3)):
	model = Sequential()
	model.add(Lambda(lambda x : (x/255.0)-0.5, input_shape=input_shape))
	model.add(Cropping2D(cropping=((crop_top,crop_bottom),(0,0))))
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	print(model.summary())
	return model

# Main code
X_train_1, y_train_1 = get_training_data(parent_dir_1)
X_train_2, y_train_2 = get_training_data(parent_dir_2)
X_train = np.concatenate((X_train_1, X_train_2), axis=0)
y_train = np.concatenate((y_train_1, y_train_2), axis=0)
model = get_lenet_model((160,320,3))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=1000)
model.save('model.h5')
