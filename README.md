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
In this competition we are classifying 104 types of flowers based on their images drawn from different public datasets. The images provided are of 4 different size:
i.	192x192
ii.	224x224
iii.	331x331
iv.	512x512
Some classes are very narrow, containing only a sub-type of flower (e.g. pink primroses) while other classes contain many sub-types (e.g. wild roses).

### b.	Data Format
The images provided are in TFRecord format. The TFRecord format is a container format frequently used in Tensorflow to group and shard data files for optimal training performance.
