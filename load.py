from collections import defaultdict

import os
import shutil
import glob

import numpy as np

from skimage import io

import cv2


def load_raw_image_collection(images_dir, target_size=(720, 1080)):
    """
    Load images form given folder.
    
    images_dir:
        str
        Path to the folder with images (frames)
    
    return:
        list
        List of resized RGB-images
    """
    images_list = glob.glob(f'{images_dir}/*.JPG')
    
    _temp = io.imread(images_list[0])
    orig_size = (_temp.shape[0], _temp.shape[1])
    scale_factor = np.mean([orig_size[0] / target_size[0], orig_size[1] / target_size[1]])
                            
    print('Original size (y, x): ', orig_size)
    print('Target size (y, x): ', target_size)
    print('Scale factor (FOCAL LENGTH is needed to be DIVIDED BY IT):', scale_factor)

    image_collection_raw = [
        cv2.resize(io.imread(filename), (target_size[1], target_size[0]))
        for filename in images_list
    ]
    
    return image_collection_raw
