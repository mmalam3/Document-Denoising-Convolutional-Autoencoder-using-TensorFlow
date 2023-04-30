# include paths for modules to import
import sys
sys.path.append('./config/')
sys.path.append('./utils/')
sys.path.append('../src/')
sys.path.append('../models/')

# import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

# importing variables
from config import DATA_DIR, RAW_DATA_DIR

# importing methods
# from utils import check_image_width_height, resize_images
from data_processing import check_image_dimensions
from data_processing import process_image
from network_model import model
from evaluate_model import evaluate_model


# ***********************************************************************
# **************** Prepare datasets for the network *********************
# ***********************************************************************

# set directories for raw 'train', 'test', and 'train_cleaned' images
raw_train_dir = os.path.join(RAW_DATA_DIR, 'train')
raw_test_dir = os.path.join(RAW_DATA_DIR, 'test')
raw_train_cleaned_dir = os.path.join(RAW_DATA_DIR, 'train_cleaned')

# get the minimum widths of heights of 'train' images
min_width, min_height = check_image_dimensions(raw_train_dir)
print(f'Minimum width: {min_width}, minimum height: {min_height}')

train_processed = []
test_processed = []
train_cleaned_processed = []

# process 'train' images by calling the process_image() method
for img in os.listdir(raw_train_dir):
    img_path = os.path.join(raw_train_dir, img)
    train_processed.append(process_image(img_path, min_height, min_width))

# process 'test' images by calling the process_image() method
for img in os.listdir(raw_test_dir):
    img_path = os.path.join(raw_test_dir, img)
    test_processed.append(process_image(img_path, min_height, min_width))

# process 'train' images by calling the process_image() method
for img in os.listdir(raw_train_cleaned_dir):
    img_path = os.path.join(raw_train_cleaned_dir, img)
    train_cleaned_processed.append(process_image(img_path, min_height, min_width))


# ***********************************************************************
# *************************** Data splitting ****************************
# ***********************************************************************

# convert list to numpy array
X_train = np.asarray(train_processed)
Y_train = np.asarray(train_cleaned_processed)
X_test = np.asarray(test_processed)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)


# ***********************************************************************
# ****************** Model Generation and Training **********************
# ***********************************************************************

# generate the model by calling the model() method
cae_model = model(min_width, min_height)
print(f'\nModel summary: {cae_model.summary()}')

# the cae_model already comes as a compiled model

# callback method
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

# start training
history = cae_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200, batch_size=16, callbacks=[callback])


# ***********************************************************************
# ************************ Model Evaluation *****************************
# ***********************************************************************

# Check how LOSS & MAE decreases over epochs
evaluate_model(history)

# predict test images / denoise test images
Y_test = cae_model.predict(X_test, batch_size=16)

# Compare noisy and denoised test images
plt.figure(figsize=(15, 25))
for i in range(0, 8, 2):
    plt.subplot(4, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i], cmap='gray')
    plt.title('Original noisy image')

    plt.subplot(4, 2, i + 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Y_test[i], cmap='gray')
    plt.title('Denoised by Convolutional Autoencoder')

plt.savefig('../reports/noise_vs_denoised_images.png')
plt.show()


# ***********************************************************************
# **************************** Model Saving *****************************
# ***********************************************************************

# Save the entire model as a SavedModel.
# os.mkdir('../saved_model')
cae_model.save('../saved_model/denoised_autoencoder_trained')
