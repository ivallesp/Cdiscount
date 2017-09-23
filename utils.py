
__author__ = "ivallesp"

import json
import numpy as np
import random
import warnings
import os
import codecs

import matplotlib.pyplot as plt

from scipy.misc import imread


def batch_generator(jl: list, data_path: str, batch_size: int, random_seed: int=655321):
    # Set seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Generate shuffle indice and initialize lists
    shuffled_indices = list(range(len(jl)))
    random.shuffle(shuffled_indices)
    image_paths, labels, ids = [], [], []

    # Generate lists of image paths, labels and ids
    for index in shuffled_indices:
        element = jl[index]
        
        for image_filename in element["filenames"]:
            image_path = os.path.join(data_path, image_filename)
            image_paths.append(image_path)
            labels.append(element.get("category_id"))
            ids.append(element["_id"])    
        
    assert len(image_paths) == len(labels) == len(ids)
    
    left_index = 0
    right_index = batch_size
    
    while right_index < len(image_paths)+batch_size: # Batches generator loop
        # Get batches
        ids_batch = ids[left_index:right_index]
        images_batch = list(map(lambda image_path: imread(image_path), image_paths[left_index:right_index]))
        labels_batch = labels[left_index:right_index]
        assert len(ids_batch) == len(images_batch) == len(labels_batch)
        
        # Batches to Numpy Arrays
        images_batch = np.row_stack(list(map(lambda img: np.expand_dims(img, 0), images_batch)))
        labels_batch = np.array(labels_batch)
        assert images_batch.shape[0] == labels_batch.shape[0]
        
        # Image transformations
        images_batch = images_batch/255.0
        
        # Update indices
        left_index += batch_size
        right_index += batch_size
        assert (right_index-left_index) <= batch_size
        
        # Yield batch
        yield(ids_batch, images_batch, labels_batch)
        
        
def make_array_of_images(images, max_images=225):
    side_size = min(int(np.sqrt(len(images))), int(np.sqrt(max_images)))
    resulting_image = None
    i = 0
    for i in range(side_size): # Join columns of images
        column_image=None
        for j in range(side_size): # Build columns of images
            img = images[i*side_size+j-side_size]
            column_image = np.row_stack([column_image, img]) if column_image is not None else img
        resulting_image = np.column_stack([resulting_image, column_image]) if resulting_image is not None else column_image
    return(resulting_image)


def imshow(image, size=[10,10]):
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.show()
