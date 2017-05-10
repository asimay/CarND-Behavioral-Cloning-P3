# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/module.png "Model Visualization"
[image2]: ./examples/center.jpg  "center img"
[image3]: ./examples/left.jpg "left Image"
[image4]: ./examples/right.jpg "right Image"
[image5]: ./examples/screenshot.png "data in excel"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Nvidia's End to End Learning for Self-Driving Cars architecture was implemented in Keras, this architecture is also known as pilotNet.

![alt text][image1]


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 184-188) 

The model includes RELU layers to introduce nonlinearity (code line 184), and the data is normalized in the model using a Keras lambda layer (code line 180). 
```sh
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1, input_shape=image_shape, output_shape=image_shape))  #(x/255.0) - 0.5
model.add(Cropping2D(cropping=((70, 25),(0, 0))))

# pilotNet Network Module
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

```sh
model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of data:

1. images from udacity dataset. (data total number is :8000)

2. 2 more laps of clock orientation's data, collected by myself used keyboard.

3. 1 laps of counter-clock orientation's data.

4. recovery data from side ways to center of road.

Data collection example: (center, left, right)

![center][image2]  ![left][image3]  ![alt right][image4]

After above all data collection, the total number of data could reach to about 14000. These data don't include augmented data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Data collecting Approach

The overall strategy for deriving a model architecture was to refer to "End to End Learning for Self-Driving Cars" by Nvidia, it is a great place to start. From the paper, data collection is the first important part. Per project requirement, data collection can only performed on Track 1.  I drove about 3 laps around Track 1 by keyboard control to collect data, and combined with Udacity sample data as starting point.

After recording and save data, the simulator saves all the frame images in IMG folder and produces a driving_log.csv file which containts all the information needed for data preparation such as path to images folder, steering angle at each frame, throttle, brake and speed values.

screen shot as below:
![data_in_excel][image5]

In this project, we only need to predict steering angle. So we will ignore throttle, brake and speed information.

#### 2. Data Augmentation and preprocessing Approach

From the observation in both tracks, there are many factor of road condition and environment to account for. Below are argumentation methods:

1. use multiple cameras, take left and right image into training dataset, also include the left/right steering angle dataset, create adjusted steering measurements for the side camera images.

2. brightness augmentation, use RGB -> HSV, and add some random noise in random image.

3. flip images to augmentation data, random flip some image for good training.

4. data preprocessing, include image normalizetion method.

5. cropping images in NN network. 
The cameras in the simulator capture 160 pixel by 320 pixel images.
Not all of these pixels contain useful information, however. In the image above, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car.
So crop it, the original image size (160x320), after cropping 70px on top and 25px on the bottom, new image size is (65x320).

6. collecting more data.

```sh
# brightness augmentation.
augment = .25+np.random.uniform(low=0.0, high=1.0)

# flip images
img = cv2.flip(img, 1)
angle = -angle

# normalization
Lambda(lambda x: (x/127.5) - 1, input_shape=image_shape, output_shape=image_shape)

# create adjusted steering measurements for the side camera images
correction = 0.2 # this is a parameter to tune
steering_left = steering_center + correction
steering_right = steering_center - correction
```


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

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
