#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Following files have been included in the submission. 
- model.py : This file contains the python code to read the images to generate the training data, augment the training data, build a training model, and save it in model.h5
- drive.py : This file is used for driving the car in autonomous mode in the simulator. This is provided [Udacity](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/drive.py). I did not need to make any changes
- model.h5 : This is a generated file that has all the information about the trained network.
- writeup_report.md : This report

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file; the car can be driven autonomously around the track by executing

```
Python drive.py model.h5
```

#### 3. Submission code is usable and readable

model.py contains all the code required to train the model. This code has been tested with several data inputs and has found to be robust and usable. Variable names are easily readable and appropriate comments have been added

Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
Lenet model that I used in Project 2 has been used as a starting point. This models a convolutional neural network with 2 layers of 2D convolution.
Each convolution layer is supported by a relu layer and a max-pooling layer. The convolution layers are followed by a flatten layer, which in turn is followed by three dense layers.
The model has been enhanced further by adding a lambda layer upfront to normalize the input data by mapping values from 0-255 to values from -0.5 to 0.5
Also, a 2D cropping layer has been added to remove crop\_top=70 rows of pixels from the top and crop\_bottom=20 pixels from the bottom

Training was performed using the keras model fit algorithm. Training was done by shuffling the data, and a batch size of 1000 was used. 
Adam classifier was used and mean square error was used to represent error. Training was done for 7 epochs

With my training data and the Lenet model, I found that I was able to achieve very good performance

#### 2. Attempts to reduce overfitting in the model

I used data from two different sources - one provided by Udacity and one that I generated using the simulator. I drove only two laps to capture data in order to avoid to over fitting. I also used different styles of driving while collecting data

#### 3. Model parameter tuning

Learning rate was tuned internally. I tried various batch sizes and found that 1000 is a good number (total number of training data images is around 150000)
Default parameters in the Lenet model generated good results

#### 4. Appropriate training data

I used data from two different sources - one provided by Udacity and one that I generated using the simulator. 
I drove only two laps to capture data in order to avoid to over fitting. 
I also used different styles of driving while collecting data

Model Architecture and Training Strategy

#### 1. Overview

As a first step, I uses a simple neural network model as proposed in lectures. This model was used to build the overall pipeline. 
Only center images from Udacity provided database with no data augmentation was used. Results were disastrous. Steering angle veered between +25 and -25 degrees and the car went around in circles.

Following steps were then taken to improve results:
1. The simple NN was replaced by a Lenet CNN. Details are mentioned in the next section
2. Additional input data was obtained by using the simulator provided
3. Data was augmented by adding horizontally flipped versions. Also, I used data from all three cameras
4. Various training parameters were used and the best combination was derived upon

I was now able to get the car to drive safely for more than a single lap. There was one instance when the car touches the shoulder, but the model is able to correct itself.

#### 2. Final Model Architecture

The final model architecture (get\_Lenet\_model) consisted of a convolution neural network with the following layers and layer sizes.
This report was obtained using print(model.summary())
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 66, 316, 6)    456         cropping2d_1[0][0]               
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 33, 158, 6)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 29, 154, 6)    906         maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 14, 77, 6)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6468)          0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 120)           776280      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 84)            10164       dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             85          dense_2[0][0]                    
====================================================================================================
Total params: 787,891
Trainable params: 787,891
Non-trainable params: 0
____________________________________________________________________________________________________
```



![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
