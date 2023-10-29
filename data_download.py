# Import required packages
import boto3
from pathlib import Path
from botocore import UNSIGNED
from botocore.client import Config
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Create an S3 client
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Define the S3 bucket name and prefix
bucket_name = 'eyes-on-the-ground'


# Function to list files and folders in an S3 bucket
def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**updated_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders


# Function to download files from S3 bucket
def download_files(s3_client, bucket_name, local_path, file_names, folders):
    local_path = Path(local_path)

    for folder in tqdm(folders):
        folder_path = Path.joinpath(local_path, folder)
        # Create all folders in the path
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(file_names):
        file_path = Path.joinpath(local_path, file_name)
        # Create a folder for the parent directory
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )


# Get the list of files and folders in the S3 bucket
file_names, folders = get_file_folders(client, bucket_name)

# Download the files to a local directory
download_files(client, bucket_name, "project_files/dataset", file_names, folders)

# Read the CSV file
df = pd.read_csv('project_files/dataset/train.csv')

# Split the dataset into train and validation sets
train, validation = train_test_split(df, test_size=0.2, stratify=df['extent'], random_state=42)
test = pd.read_csv('project_files/dataset/test.csv')


# Define functions to read and process images
def read_single_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return np.array(image)


def read_all_images(directory):
    images = np.empty((0, 224, 224, 3)).astype(np.uint8)
    for file_name in os.listdir(directory):
        image_path = Path.joinpath(directory, file_name)
        images = np.append(images, read_single_image(image_path), axis=0)

    print('All images loaded successfully!')
    return images


# Read and process images
images = read_all_images('project_files/dataset/train')
test_images = read_all_images('project_files/dataset/test')

# Create masks for train and validation datasets
train_mask = df['ID'].isin(train['ID'])
validation_mask = df['ID'].isin(validation['ID'])

# Filter train and validation images
train_images = images[train_mask]
validation_images = images[validation_mask]

# Save the processed images as .npy files
np.save('project_files/dataset/train.npy', train_images)
np.save('project_files/dataset/validation.npy', validation_images)
np.save('project_files/dataset/test.npy', test_images)
