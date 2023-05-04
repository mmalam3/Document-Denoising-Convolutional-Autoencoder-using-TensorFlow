# Document Denoising Convolutional Autoencoder using Tensorflow

This repository contains the implementation of a Denoising Convolutional Autoencoder (CAE) using TensorFlow, Keras, and Python. The goal of this project is to perform noise reduction in noisy documents, such as scanned documents, or images of documents.

The autoencoder architecture used in this project is a **Convolutional Neural Network (CNN)**. It consists of two components:
1. An **encoder** that takes a noisy document as input and encodes it into a low-dimensional representation, and
2. A **decoder** that takes the low-dimensional representation outputted by the encoder and reconstructs the original document discarding the noise.

## Dataset

The [denoising-dirty-documents](https://www.kaggle.com/competitions/denoising-dirty-documents/data) dataset is used in this project for training and testing the models. The dataset provides images of documents containing various style of texts. It has three sets of data:

1. __train__ data: images of documents used for training the model to which synthetic noise has been added to simulate real-world, messy artifacts, 
2. __train_cleaned__ data: dataset with denoised __train__ data used for validation during the training procedure, and
3. __test__ data: noisy images of documents to be used for testing the mode.

## Usage

To run this project, 

