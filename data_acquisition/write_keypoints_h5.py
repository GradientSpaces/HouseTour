import h5py
import os
from PIL import Image, ExifTags
import numpy as np
from tqdm import tqdm
import argparse


keypoint_h5   = h5py.File('keypoints.h5', 'w')

images = os.listdir('../keyframes_resized/')
images = sorted(images)

width, height = 512, 288    

# We want to add each pixel coordinate as a keypoint since Mast3r is not a keypoint-based model
x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
pixel_indices = np.stack([x_indices, y_indices], axis=-1).reshape(-1, 2).astype(np.float16)
pixel_indices += 0.5

# print(f"Pixel indices shape: {pixel_indices.reshape(-1, 2)}")

for image in tqdm(images):
    fname = image.split('.')[0]
    keypoint_h5.create_dataset(fname, data=pixel_indices.reshape(-1, 2))
