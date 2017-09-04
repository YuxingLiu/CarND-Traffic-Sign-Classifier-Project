# CarND-Traffic-Sign-Classifier-Project

This repository presents the code to train a deep neural network for Classification of German Traffic Signs.


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

[//]: # (Image References)

[image1]: ./figures/all_signs.png "Visualization of Traffic Signs"
[image2]: ./figures/distribution_traffic_signs.png "Distribution of Traffic Signs"
[image3]: ./figures/random_noise.jpg "Random Noise"
[image4]: ./figures/placeholder.png "Traffic Sign 1"
[image5]: ./figures/placeholder.png "Traffic Sign 2"
[image6]: ./figures/placeholder.png "Traffic Sign 3"
[image7]: ./figures/placeholder.png "Traffic Sign 4"
[image8]: ./figures/placeholder.png "Traffic Sign 5"

---

## Step 1: Data Set Summary & Exploration

### Load the data

Download the pickled [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip), in which the images have been resized to 32x32.

```python
import pickle
import os

training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

### Provide a basic summary of the data set using python, numpy and/or pandas

The numpy library are used to calculate summary statistics of the traffic signs data set:

```python
import numpy as np

n_train = len(y_train)
n_validation = len(y_valid)
n_test = len(y_test)
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))
```

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image1]

The mappings from the class ID to the actual sign name can be found in ['signnames.csv'](https://github.com/YuxingLiu/CarND-Traffic-Sign-Classifier-Project/blob/master/signnames.csv). The count of each sign in each data set is calculated and shown as follows:

```python
signs_num_train = np.array([sum(y_train == i) for i in range(len(np.unique(y_train)))])
signs_num_valid = np.array([sum(y_valid == i) for i in range(len(np.unique(y_valid)))])
signs_num_test = np.array([sum(y_test == i) for i in range(len(np.unique(y_test)))])
```
| ClassId   |   SignName	        	    |   NumTrain    |   NumValid    |   NumTest |
|:---------:|:-----------------------------:|:-------------:|:-------------:|:---------:| 
| 0         |   Speed limit (20km/h)        |   180         |   30          |   60      |
| 1         |   Speed limit (30km/h)        |   1980        |   240         |   720     |
| 2         |   Speed limit (50km/h)        |   2010        |   240         |   750     |
| 3         |   Speed limit (60km/h)        |   1260        |   150         |   450     |
| 4         |   Speed limit (70km/h)        |   1770        |   210         |   660     |
| 5         |   Speed limit (80km/h)        |   1650        |   210         |   630     |
| 6         |   End of speed limit (80km/h) |   360         |   60          |   150     |
| 7         |   Speed limit (100km/h)       |   1290        |   150         |   450     |
| 8         |   Speed limit (120km/h)       |   1260        |   150         |   450     |
| 9         |   No passing                  |   1320        |   150         |   480     |
| 10        |   No passing for vehicles over 3.5 metric tons | 1800 | 210   |   660     |
| 11        |   Right-of-way at the next intersection   | 1170    | 150     |   420     |
| 12        |   Priority road               |   1890        |   210         |   690     |
| 13        |   Yield                       |   1920        |   240         |   720     |
| 14        |   Stop                        |   690         |   90          |   270     |
| 15        |   No vehicles                 |   540         |   90          |   210     |
| 16        |   Vehicles over 3.5 metric tons prohibited | 360  |   60      |   150     |
| 17        |   No entry                    |   990         |   120         |   360     |
| 18        |   General caution             |   1080        |   120         |   390     |
| 19        |   Dangerous curve to the left |   180         |   30          |   60      |
| 20        |   Dangerous curve to the right|   300         |   60          |   90      |
| 21        |   Double curve                |   270         |   60          |   90      |
| 22        |   Bumpy road                  |   330         |   60          |   90      |
| 23        |   Slippery road               |   450         |   60          |   150     |
| 24        |   Road narrows on the right   |   240         |   30          |   90      |
| 25        |   Road work                   |   1350        |   150         |   480     |
| 26        |   Traffic signals             |   540         |   60          |   180     |
| 27        |   Pedestrians                 |   210         |   30          |   60      |
| 28        |   Children crossing           |   480         |   60          |   150     |
| 29        |   Bicycles crossing           |   240         |   30          |   90      |
| 30        |   Beware of ice/snow          |   390         |   60          |   150     |
| 31        |   Wild animals crossing       |   690         |   90          |   270     |
| 32        |   End of all speed and passing limits |   210 |   30          |   60      |
| 33        |   Turn right ahead            |   599         |   90          |   210     |
| 34        |   Turn left ahead             |   360         |   60          |   120     |
| 35        |   Ahead only                  |   1080        |   120         |   390     |
| 36        |   Go straight or right        |   330         |   60          |   120     |
| 37        |   Go straight or left         |   180         |   30          |   60      |
| 38        |   Keep right                  |   1860        |   210         |   690     |
| 39        |   Keep left                   |   270         |   30          |   90      |
| 40        |   Roundabout mandatory        |   300         |   60          |   90      |
| 41        |   End of no passing           |   210         |   30          |   60      |
| 42        |   End of no passing by vehicles over 3.5 metric tons | 210 | 30|  90      |

The traffic signs distribution in the training set is plotted below. It can be observed that the number of each sign varies dramatically in each dataset, but their distrubution trends are similar among three datasets. It basically means some signs are expected to see more frequently than the others. Therefore, the relative ratio of signs is kept unchanged when augmenting the data.

![alt text][image2]



## Step 2: Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				    |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 				    |
| Fully connected		| outputs 256        						    |
| RELU					|												|
| Dropout		        | keep_prob = 0.5							    |
| Fully connected		| outputs 128        						    |
| RELU					|												|
| Dropout		        | keep_prob = 0.5							    |
| Fully connected		| outputs 43        						    |
| RELU					|												|
| Softmax				| probabilities of 43 traffic signs             |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


