# include paths for modules to import
import sys
sys.path.append('../convolutional_autoencoder/config/')

# import necessary libraries
import os
import cv2
import numpy as np
from config import RAW_DATA_DIR


# method to get minimum widths and heights of images
def check_image_dimensions(image_dir):
    width, height = [], []

    for image_name in os.listdir(image_dir):
        # read the current image
        img = cv2.imread(os.path.join(image_dir, image_name))

        # get width and height of the current image
        height_, width_ = img.shape[0], img.shape[1]

        # add height and width into lists
        height.append(height_)
        width.append(width_)

    return min(width), min(height)


# method to process images
def process_image(img_path, new_height, new_width):
    # Open the image file
    image = cv2.imread(img_path)

    # convert the image intp a numpy array
    image = np.asarray(image, dtype="float32")

    # Resize image to the fixed new size
    image = cv2.resize(image, (new_width, new_height))

    # Normalize image to the range 0-255
    image = image / 255.0

    # reshape the dimension
    image = np.reshape(image, (new_height, new_width, 3))

    return image
