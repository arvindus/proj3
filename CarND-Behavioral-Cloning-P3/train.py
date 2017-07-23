import csv
import cv2
import numpy as np

parent_dir = '../data/'
steering_correction_factor = 0.2
def augment_data(images, measurements):
	augmented_images = []
	augmented_measurements = []
	for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_images.append(cv2.flip(image,1))
		augmented_measurements.append(measurement)
		augmented_measurements.append(measurement*-1.0)
	
	print('augment step 1 done...')
    
	return augmented_images, augmented_measurements
    
  
def get_training_data(parent_dir):
	data_dir = parent_dir + 'IMG/'
	csv_file = parent_dir + 'driving_log.csv'
	images = []
	measurements = []
	print('csv file', csv_file)
	count = 0
	with open(csv_file) as file:
		reader = csv.reader(file)
		for line in reader:
			count = count + 1
			print('count', count)
			if line[0] == 'center':
				continue
			fname = data_dir + line[0].split('/')[-1]
			#print(fname)
			image = cv2.imread(fname)
			image = image[...,::-1] #bgr to rgb
			images.append(image)
			measurements.append(float(line[3]))
  
	print('reading done...')
	# Data augmentation
	images, measurements = augment_data(images, measurements)
	print('augment done...')
	X_train = np.array(images)
	y_train= np.array(measurements)
	return X_train, y_train

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Flatten, Dense
def get_basic_model(input_shape=(160,320,3)):
	model = Sequential()
	model.add(Lambda(lambda x : (x/255.0)-0.5, input_shape=input_shape))
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(1))
	return model
	
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
def get_lenet_model(input_shape=(160,320,3)):
	model = Sequential()
	model.add(Lambda(lambda x : (x/255.0)-0.5, input_shape=input_shape))
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model

X_train, y_train = get_training_data(parent_dir)
model = get_lenet_model((160,320,3))
model.compile(loss='mse', optimizer='adam')
print('compile done...')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
print('train done..')
model.save('model.h5')
		
		