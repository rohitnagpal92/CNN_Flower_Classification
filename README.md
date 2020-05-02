# FLOWER CLASSIFICATION USING TPU
The objtive of this post is to build a deep model CNN model that identifies the type of flowers in a dataset of images

## 1.	Introduction
### a.	Description
●	For our project we decided to build Flower Classification with TPU's. TPU, which stands for Tensor Processing Unit, are powerful hardware accelerators specialized in deep learning tasks.

●	TPUs were developed by Google to process large image databases, such as extracting all the text from Street View.

●	The latest Tensorflow release (TF 2.1) was focused on TPUs and they are now supported both through the Keras high-level API and at a lower level, in models using a custom training loop.

### b.	Objective

●	The objective of the model is to build a Deep Learning model to classify 104 types of flowers using TPUs. 

●	As we know, training a deep learning model with high resolution images is very time consuming and resource exhaustive, without using GPUs or TPUs, it can take up to hours to train the model even for 1 epoch whereas it could be done within seconds using GPUs and TPUs.

●	Therefore, the main objective of this competition is to make sure competitors learn to use the TPUs and analyze how the solutions are accelerated by TPU's.

### c.	Difference between CPU, GPU and TPU
Before we delve into the details regarding the classification model. Let review diidference between different processing units available:
●	CPU (Central Processing Unit):  CPU is a general-purpose processor means a CPU works with software and memory. A CPU store the calculation results on memory thus limiting the total throughput and consuming significant energy.

●	GPU (Graphics Processing Unit): To gain higher throughput than a CPU the modern GPU usually has 2,500–5,000 ALUs in a single processor that means we could execute thousands of multiplications and additions simultaneously. But, the GPU is still a general purpose processor and GPU need shared memory to read and store the intermediate calculation results.

●	TPU (Tensor Processing Unit): In case of TPU, instead of having a general-purpose processor, it has a matrix processor specialized for neural network workloads. They can handle the massive multiplications and additions for neural networks, at blazingly fast speeds while consuming much less power and inside a smaller physical footprint.


## 2.	Data Description and Data Preparation
### a.	Dataset Description
In this competition we are classifying 104 types of flowers based on their images drawn from different public datasets. The images provided are of 2 different size: (331x331) and (512x512).

Some classes are very narrow, containing only a sub-type of flower (e.g. pink primroses) while other classes contain many sub-types (e.g. wild roses).

### b.	Data Format
The images provided are in TFRecord format. The TFRecord format is a container format frequently used in Tensorflow to group and shard data files for optimal training performance.

## 3.	Model Description
### I.	CNN model for input size of (331, 331, 3) and (512, 512, 3) using TPU:
Build a plain vanilla convolutional network model with following hyperparameters: 
#### Model Specification:
❖	Layers: Conv2D, BatchNormalization. Dropout and GlobalAveragePooling2D (or Flatten) 
❖	Activation function (in last layer): Softmax with 104 classes of Flowers 
❖	Loss function: sparse_categorical_crossentropy 
❖	Optimizer: Adam 
❖	Regularizer: EarlyStopping 

#### Dataset: 
❖	12753 training images, 
❖	3712 validation images, 
❖	7382 unlabeled test images 

#### Performance: 
~74-76% Validation Accuracy 

#### Time taken:
❖	size 331x331 -> ~2 - 2.5 hours 
❖	size 512x512 -> ~3 hours 

### II.	Fine tuning model using DenseNet and EfficientNet EB7 with LearningRateScheduler
#### i.	Model: Build two fine-tuned models using DenseNet and EB7.

##### Model 1: DenseNet
❖	Pre-trained Model: 
o	DenseNet201(weights='imagenet', include_top=False, input_shape=image_shape) 
o	include_top=False. We have not loaded the last two fully connected layers which act as the classifier. We are just loading the convolutional layers.
o	trained all convolutional layers (as a part of fine tuning)
❖	Layers: DenseNet201, GlobalAveragePooling2D (or Flatten) 
❖	Activation function (in last layer): Softmax with 104 classes of Flowers
❖	Loss function: sparse_categorical_crossentropy 
❖	Optimizer: Adam 
❖	Regularizer: EarlyStopping, LearningRateScheduler

##### Model 2: EfficientNetB7
❖	Pre-trained Model: 
o	efn.EfficientNetB7(weights='noisy-student', include_top=False, input_shape=image_shape) 
o	include_top=False. We have not loaded the last two fully connected layers which act as the classifier. We are just loading the convolutional layers.
o	trained all convolutional layers (as a part of fine tuning)
❖	Layers: DenseNet201, GlobalAveragePooling2D (or Flatten) 
❖	Activation function (in last layer): Softmax with 104 classes of Flowers
❖	Loss function: sparse_categorical_crossentropy 
❖	Optimizer: Adam 
❖	Regularizer: EarlyStopping, LearningRateScheduler

##### Dataset: Size 512x512 
o	For Training used sets of both Training and Validation Images to provide the model with a greater number of images - 12753 training images + 3712 validation images
o	7382 unlabeled test images 

##### Performance:  
~99-100% Training Accuracy

##### Time taken: 
❖	Less than 2 to 3 Hours

##### Prediction: 
While predicting the class of test images, we calculated probabilities from both the models individually, then took the weighted average of the probabilities and last used "argmax(probabilities, axis=-1)" to get the class with highest probability.

##### Observation: 
❖	Model took more time to converge due to low learning rate with higher epochs and having two different models
❖	But the overall model's accuracy improved significantly, and we achieved the highest rank of 54 out of 700 participating teams.








