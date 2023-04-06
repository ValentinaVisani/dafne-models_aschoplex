#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import uuid

import numpy as np
from scipy.ndimage import zoom

# Get the path of the script's parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to sys.path to import common
sys.path.append(parent_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to create")
    parser.add_argument("data_path", help="Path to data folder (containing the '*.npz' files)")
    args = parser.parse_args()

    data_path = args.data_path

    # Load all the npz files in the folder
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')]
    data_list = []
    for file in data_files:
        print(f'Loading {file}')
        npz_file = np.load(file)
        data_list.append(npz_file)

    # Find the most common resolution among all the data
    resolutions = [data['resolution'][:2] for data in data_list]
    common_resolution = np.median(resolutions, axis=0)

    # Find the most common size in voxel of all the slices
    sizes = [data['data'].shape[:2] for data in data_list]
    sizes_dict = {}
    for size in sizes:
        if size in sizes_dict:
            sizes_dict[size] += 1
        else:
            sizes_dict[size] = 1
    common_size = max(sizes_dict, key=sizes_dict.get)

    # Create a set with all the different labels present in the files
    labels = set()
    for data in data_list:
        for key in data.keys():
            if key.startswith('mask_'):
                labels.add(key[5:])

    labels = sorted(labels)

    # Crete a dictionary with indices for each label
    label_dict = {i+1: label for i, label in enumerate(labels)}

    # load the model template
    with open('generate_model.py.tmpl', 'r') as f:
        source = f.read()

    # replace the variables
    source = source.replace('%%MODEL_NAME%%', f'"{args.model_name}"')
    source = source.replace('%%MODEL_RESOLUTION%%', f'[{common_resolution[0]:.2f}, {common_resolution[1]:.2f}]')
    source = source.replace('%%MODEL_SIZE%%', f'[{common_size[0]:d}, {common_size[1]:d}]')
    # Generate a version 4 UUID
    uid = uuid.uuid4()
    # Convert UUID to string
    uid_str = str(uid)
    source = source.replace('%%MODEL_UID%%', f'"{uid_str}"')
    source = source.replace('%%LABELS_DICT%%', str(label_dict))

    # write the new model generator script
    with open(f'generate_{args.model_name}_model.py', 'w') as f:
        f.write(source)

    # remove the generate_convert import from the source code
    source = source.replace('from common import generate_convert', '')


    # replace the generate_convert function with a dummy function
    def dummy_generate_convert(*args, **kwargs):
        pass

    new_locals = {'generate_convert': dummy_generate_convert}
    # now when we execute the file, only the functions will be created. This is used to extract the make_unet function
    exec(source, globals(), new_locals)
    create_model_function = new_locals['make_unet']
    model = create_model_function()

    image_list = []

    # normalize the training data
    for data in data_list:
        resolution = np.array(data['resolution'])
        zoomFactor = resolution/common_resolution
        img_3d = data['data']
        img_3d = zoom(img_3d, [zoomFactor[0], zoomFactor[1], 1]) # resample the image to the model resolution


        img = padorcut(img, MODEL_SIZE)
        imgbc = biascorrection.biascorrection_image(img)


    # now train the model on the data




