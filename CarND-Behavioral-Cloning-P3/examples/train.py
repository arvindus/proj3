import csv
import cv2
import numpy as np
lines = []
dirname = 'mydata'
with open('..\\'+dirname+'\\driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []

for line in lines:
  if float(line[6]) < 0.1 :
     continue
  source_path = line[0]
  filename = source_path.split('\\')[-1]
  current_path = '..\\'+dirname+'\\IMG\\' + filename
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  print(measurement)
  measurements.append(measurement)
  
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=8)

model.save('model.h5')