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
[image3]: ./figures/exp_original.png "Original Images"
[image4]: ./figures/exp_aug.png "Augmented Images"
[image5]: ./figures/exp_aug_pp.png "Pre-processed Images"
[image6]: ./figures/new_pp.png "5 New Traffic Signs"
[image7]: ./figures/precision_recall.png "Precision and Recall"
[image8]: ./figures/sign1.png "Traffic Sign 1"
[image9]: ./figures/sign2.png "Traffic Sign 2"
[image10]: ./figures/sign3.png "Traffic Sign 3"
[image11]: ./figures/sign4.png "Traffic Sign 4"
[image12]: ./figures/sign5.png "Traffic Sign 5"
[image13]: ./figures/conv1_feature_map.png "Conv1 Feature Map"

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

The numpy library is used to calculate summary statistics of the traffic signs data set:

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
import pandas as pd

signs_pd = pd.read_csv('signnames.csv')
signs_num_train = np.array([sum(y_train == i) for i in range(len(np.unique(y_train)))])
signs_num_valid = np.array([sum(y_valid == i) for i in range(len(np.unique(y_valid)))])
signs_num_test = np.array([sum(y_test == i) for i in range(len(np.unique(y_test)))])
signs_pd['NumTrain'] = pd.Series(signs_num_train)
signs_pd['NumValid'] = pd.Series(signs_num_valid)
signs_pd['NumTest'] = pd.Series(signs_num_test)
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

The traffic signs distribution in the training set is plotted below. It can be observed that the number of each sign varies dramatically in each dataset, but their distrubution trends are similar among three datasets. It basically means some signs are expected to see more frequently than the others. Therefore, the relative ratio of signs is kept unchanged later in training set augmentation.

![alt text][image2]



## Step 2: Design and Test a Model Architecture

### Pre-process the Data Set

From the previous visualization, one could see that the images are collected in different lighting conditions from different distances and viewing angles. To make the classifier more robust against potential deformations, the original training set is augmented by adding 5 transformed versions, yielding 208,794 samples in total. In this study, the following perturbations are randomly applied:
* Gaussian blur ([0, 0.5] sigma, 50% of samples)
* Contrast normalization ([0.75, 1.5] ratio)
* Translation ([-2, 2] pixels)
* Scale ([0.8, 1.2] ratio)
* Rotation ([-15, 15] degrees)

The [imgaug](https://github.com/aleju/imgaug) library is used to augment images in the training set:

```python
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_px={"x": (-2, 2), "y": (-2, 2)},
        rotate=(-15, 15)
    )
], random_order=True) # apply augmenters in random order

def augment_images(X_data, y_data, N_aug=5):
    X_aug = np.zeros((X_data.shape[0]*N_aug, X_data.shape[1], X_data.shape[2], X_data.shape[3]), dtype=np.uint8)
    y_aug = np.zeros((X_data.shape[0]*N_aug), dtype=np.uint8)
    for i in range(len(X_data)):
        images = np.array([X_data[i] for _ in range(N_aug)], dtype=np.uint8)
        images_aug = seq.augment_images(images)
        X_aug[i*N_aug:(i+1)*N_aug] = images_aug
        y_aug[i*N_aug:(i+1)*N_aug] = y_data[i]
    
    return X_aug, y_aug
    
X_train_aug, y_train_aug = augment_images(X_train, y_train, N_aug=5)
X_train = np.concatenate((X_train, X_train_aug), axis=0)
y_train = np.concatenate((y_train, y_train_aug), axis=0)    
```

Next, the following preprocessing steps are applied for all the data sets:
* Histogram equalization to enhance contrast
* Normalization in range [-0.5, 0.5]

```python
import cv2

def pre_process_image(image):
    # histogram equalization of RGB image
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    # normalization in range [-0.5, 0.5]
    image = image/255.0 - 0.5
    return image

def pre_process_images(images):
    return np.array([pre_process_image(images[i]) for i in range(len(images))])
    
X_train = pre_process_images(X_train)
X_valid = pre_process_images(X_valid)
X_test = pre_process_images(X_test)    
```

Here is an example of 10 traffic sign images before and after pre-processing.

Original:
![alt text][image3]

After augmentation:
![alt text][image4]

After pre-processing:
![alt text][image5]

### Model Architecture

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
 
The first 3 convolutional layers consist of 32, 64, 128 (respectively) 5x5 filters, followed by 2x2 max pooling. The output of the third convolutional layer is flatten and fed to 3 fully connected layers, which are composed of 256, 128, 43 neurons, respectively. In addition, dropout is applied to the outputs of the 2 hidden layers.

```python
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def con2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') + b
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides = [1,k,k,1], padding="SAME")

def fc2d(x, W, b):
    x = tf.matmul(x, W) + b
    return tf.nn.relu(x)
```
```python    
### Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)  
```
```python
### Weights and Biases
mu = 0
sigma = 0.1

weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], mu, sigma)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], mu, sigma)),
    'wc3': tf.Variable(tf.truncated_normal([5, 5, 64, 128], mu, sigma)),
    'wd1': tf.Variable(tf.truncated_normal([4*4*128, 256], mu, sigma)),
    'wd2': tf.Variable(tf.truncated_normal([256, 128], mu, sigma)),
    'wd3': tf.Variable(tf.truncated_normal([128, n_classes], mu, sigma))}

biases = {
    'bc1': tf.Variable(tf.zeros(32)),
    'bc2': tf.Variable(tf.zeros(64)),
    'bc3': tf.Variable(tf.zeros(128)),
    'bd1': tf.Variable(tf.zeros(256)),
    'bd2': tf.Variable(tf.zeros(128)),
    'bd3': tf.Variable(tf.zeros(n_classes))}
```
```python
### ConvNet
# Layer 1: Convolutional. Input = 32x32x3. Output = 16x16x32.
conv1 = con2d(x, weights['wc1'], biases['bc1'])
conv1 = maxpool2d(conv1, k=2)

# Layer 2: Convolutional. Output = 8x8x64.
conv2 = con2d(conv1, weights['wc2'], biases['bc2'])
conv2 = maxpool2d(conv2, k=2)

# Layer 3: Convolutional. Output = 4x4x128.
conv3 = con2d(conv2, weights['wc3'], biases['bc3'])
conv3 = maxpool2d(conv3, k=2)

# Layer 4: Fully Connected. Input = 1024. Output = 256.
fc1 = fc2d(flatten(conv3), weights['wd1'], biases['bd1'])
fc1 = tf.nn.dropout(fc1, keep_prob)

# Layer 5: Fully Connected. Input = 256. Output = 128.
fc2 = fc2d(fc1, weights['wd2'], biases['bd2'])
fc2 = tf.nn.dropout(fc2, keep_prob)

# Layer 6: Fully Connected. Input = 128. Output = 43.
logits = fc2d(fc2, weights['wd3'], biases['bd3'])
```

### Train, Validate and Test the Model

To train the model, the parameters are assigned as follows:
```python
EPOCHS = 25
BATCH_SIZE = 128
learning_rate = 0.0001
dropout = 0.5  # Dropout, probability to keep units
```

The optimization problem considers to minimize the average cross entropy, using the [Adam optimizer](https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/train/AdamOptimizer). The training pipeline is obtained as follows:
```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_operation
```

To monitor the progress of training as well as get the prediction accuracy of the model, an evaluation function is defined:
```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

Then, the training is conducted by using the following code:
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        # rewrite due to memory error
        # X_train, y_train = shuffle(X_train, y_train)
        idx_rand = shuffle(np.arange(len(y_train)))
        
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            # rewrite due to memory error            
            #batch_x, batch_y = X_train[offset:end|], y_train[offset:end]
            batch_x, batch_y = X_train[idx_rand[offset:end]], y_train[idx_rand[offset:end]]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {0:3},    Validation Accuracy = {1:.3f}".format(i+1, validation_accuracy))
        
    saver.save(sess, 'saved_models/ConvNet.ckpt')
    print("Model saved")
```

The final model results were:
* training set accuracy of 99.0%
* validation set accuracy of 98.4% 
* test set accuracy of 96.7%

It's worth mentioning that it is an iterative process to find a solution with 98.4% validation set accuracy. The first architecture that was tried is the [LeNet-5](http://yann.lecun.com/exdb/lenet/) convolutional network for handwritten digit recognition. On the normalized training set (without augmentation and preprocessing), the initial architecture yielded about 91% accuracy. Converting the RGB images to grayscale didn't show improvement on the LeNet-5, so color images are used in the study.

As illustrated in the dataset visualization, the real-world variabilities such as viewpoint, lighting conditions, motion-blur, sun glare, colors fading and low resolution pose difficulties for traffic sign classification. To this extent, data augmentation and Histogram equalization were utilized to enhance the robustness of the model against small disturbances. 

After implementing augmentation and preprocessing, the validation accuracy of LeNet-5 was improved to 94%, while the training accuracy could reach 98%. The gap between the two accuracies implies overfitting, hence two dropout layers with 'keep_prob=0.5' were added to the fully connected hidden layers. Meanwhile, to further improve the performance of the classifier as well as to generate some 'redundant' features required for the dropout technique, a deeper (one additional convolutional layer) and wider (more features in each layer) network architecture was constructed.

As regards the parameters tuning, the 'EPOCHS' and 'BATCH_SIZE' were limited by the computational power and memory size, respectively. With default 'learning_rate=1e-3', it was found that the validation accuracy could reach 97% after 3 iterations, and then oscillated between 97% and 98%. Therefore, 'learning_rate=1e-4' was selected such that the validation accuracy progressively increased after each epoch.


## Step 3: Test the Model on New Images

### Load and Output the Images


####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here is the [five German traffic signs](https://github.com/YuxingLiu/CarND-Traffic-Sign-Classifier-Project/tree/master/new_images) found on the web, which are down-sampled to 32x32 and pre-processed.

```python
name_images = os.listdir("new_images/")
X_web = np.zeros((len(name_images), 32, 32, 3), dtype=np.float64)
y_web = np.zeros(len(name_images), dtype=np.uint8)

for i in range(len(name_images)):
    # Read in the image
    image = cv2.imread('new_images/'+name_images[i])
    # Brighness normalization
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])  
    # Resize
    resized = cv2.resize(img_yuv, (32,32), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(resized, cv2.COLOR_YUV2RGB)
    # normalization
    X_web[i] = image/255.0 - 0.5
    # assign labels
    y_web[i] = np.array(name_images[i][-6:-4]).astype(np.uint8)
```

![alt text][image6] 


### Predict the Sign Type for Each Image

```python
with tf.Session() as sess:
    saver.restore(sess, 'saved_models/ConvNet.ckpt')
        
    y_pred = sess.run(tf.argmax(logits, 1), feed_dict={x: X_web, y: y_web, keep_prob: 1})
    accuracy = evaluate(X_web, y_web)
```

Here are the results of the prediction:

| ClassId   |   Image       |   Prediction      |   Probability | 
|:---------:|:-------------:|:-----------------:|:-------------:| 
| 21        | Double Curve  | Double Curve      |   98.7%       |
| 12        | Priority Road | Priority Road     |   100%        |
| 25        | Road Work     | Road Work         |   100%        |
| 23        | Slippery Road | Slippery Road     |   97.7%       |
| 2         | Speed Limit (50km/h)  |   Speed Limit (50km/h)    | 99.9% |

The model was able to correctly guess 5 of the 5 traffic signs with over 97% certainty, which gives an accuracy of 100%. To further evaluate the performance on individual sign types, the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) on the test set are calculated as follows:

```python
def precision_recall(y_data, y_pred):
    num_SignId = len(np.unique(y_data))
    pr_data = np.zeros([num_SignId,2])
    for i in range(num_SignId):
        true_obsv = np.equal(y_pred, i)
        true_actual = np.equal(y_data, i)
        true_pos = np.logical_and(true_obsv, true_actual)
        num_obsv = np.sum(true_obsv.astype(np.float32))
        num_actual = np.sum(true_actual.astype(np.float32))
        num_pos = np.sum(true_pos.astype(np.float32))
        pr_data[i] = [num_pos/num_obsv, num_pos/num_actual]
        
    return pr_data

with tf.Session() as sess:
    saver.restore(sess, 'saved_models/ConvNet.ckpt')
        
    y_pred = sess.run(tf.argmax(logits, 1), feed_dict={x: X_test, y: y_test, keep_prob: 1})
    pr_test = precision_recall(y_test, y_pred)
```

![alt text][image7]

As shown in the above figure, the precision and recall of those 5 classes are above 80%, which justifies the high accuracy on the 5 new images. On the other hand, the recall values of class 20 and 27 are relatively low, which gives some insight into how to better augment the data set and how to fine tune the model.

### Output Top 5 Softmax Probabilities For Each Image

```python
prob_softmax = tf.nn.softmax(logits)
pred_top5 = tf.nn.top_k(prob_softmax, k=5)

with tf.Session() as sess:
    saver.restore(sess, 'saved_models/ConvNet.ckpt')
    
    predictions_top5 = sess.run(pred_top5, feed_dict={x: X_web, y: y_web, keep_prob: 1})
    
    for i in range(len(y_web)):
        plt.figure(figsize = (8,3))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,3])
        ax = plt.subplot(gs[0])
        ax.set_aspect('equal')
        plt.imshow(X_web[i]+.5)
        plt.text(1,2,str(y_web[i]),color='k',backgroundcolor='y')
        plt.axis('off')

        plt.subplot(gs[1])
        plt.barh(5-np.arange(5), predictions_top5.values[i], align='center')
        plt.xlim(xmax=1)
        for j in range(5):
            plt.text(predictions_top5.values[i][j]+0.02, 4.9-j,
                     str(predictions_top5.indices[i][j])+': '
                     +signs_pd['SignName'][predictions_top5.indices[i][j]])
        plt.axis('off')
```

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]


## Step 4: Visualizing the Neural Network's State with Test Images

Here are the first convolutional layer's feature maps in response to the first new image from the web (21, double curve). It can be seen that each feature map focuses on a certain portion of the image. For instance, feature maps 6, 11 and 27 react with high activation to the trianglar outline of the sign, while feature maps 15 and 26 roughly capture the z-shaped symbol.

![alt text][image13]
