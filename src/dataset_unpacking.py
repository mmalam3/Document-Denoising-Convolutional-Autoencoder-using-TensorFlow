# include paths for modules to import
import sys
sys.path.append('../convolutional_autoencoder/config/')

# import necessary libraries
import os
from zipfile import ZipFile
from config import DATA_DIR, RAW_DATA_DIR


# Download the "denoising-dirty-documents" dataset from this URL:
# https://www.kaggle.com/competitions/denoising-dirty-documents/data
# and keep the downloaded dataset into the project directory

# unzip the downloaded dataset into the working directory
with ZipFile('../denoising-dirty-documents.zip', 'r') as f:
    f.extractall('../data/')

# unzip individual dataset files from the 'data' directory and keep them in 'raw' directory
for file in os.listdir(DATA_DIR):
    # set the path of the compressed file
    file_path = os.path.join(DATA_DIR, file)

    with ZipFile(file_path, 'r') as f:
        # extract into the 'raw' directory
        f.extractall(RAW_DATA_DIR)

    # remove the just compressed file
    os.remove(file_path)






