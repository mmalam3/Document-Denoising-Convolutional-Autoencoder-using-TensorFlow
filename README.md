# Document Denoising Convolutional Autoencoder using Tensorflow

This repository contains the implementation of a Denoising Convolutional Autoencoder (CAE) using `TensorFlow`, `OpenCV`, `Keras`, `Scikit-Learn`, and `Python`. The goal of this project is to perform noise reduction in noisy documents, such as scanned documents, or images of documents.

The autoencoder architecture used in this project is a **Convolutional Neural Network (CNN)**. It consists of two components:
1. An **encoder** that takes a noisy document as input and encodes it into a low-dimensional representation, and
2. A **decoder** that takes the low-dimensional representation outputted by the encoder and reconstructs the original document discarding the noise.

## Dataset

The [denoising-dirty-documents](https://www.kaggle.com/competitions/denoising-dirty-documents/data) dataset is used in this project for training and testing the models. The dataset provides images of documents containing various style of texts. It has three sets of data:

1. __train__ data: images of documents used for training the model to which synthetic noise has been added to simulate real-world, messy artifacts, 
2. __train_cleaned__ data: dataset with denoised __train__ data used for validation during the training procedure, and
3. __test__ data: noisy images of documents to be used for testing the mode.

## Usage

You can run this project either 1) in Colab, or 2) in our own machine installing `TensorFlow`, `cv2`, and `scikit-learn`.  

1. To run the project in **Google Colab**, you need to open the [denoising_convolutional_autoencoder.ipynb](https://github.com/kayanmorshed/Document-Denoising-Convolutional-Autoencoder-using-Tensorflow/tree/main/notebooks) file from the `notebooks` directory. The notebook contains all the required codes along with suitable comments. Since the datate is hosted in **Kaggle**, the detailed instructions of how to download and preprocess the dataset correctly are also included in the notebook.    

2. To run the project in your own machine, use the following commands to install necessary tool/libraries:

```
python -m pip install -U pip # to install pip
pip install tensorflow
pip install pip install opencv-python
pip install -U scikit-learn
```

This project also uses two widely used Python libraries: `numpy` and `matplotlib`. If your machine doesn't have these libraries included in your Python, use the following commands to install them: 
```
pip install numpy
python -m pip install -U matplotlib
```
