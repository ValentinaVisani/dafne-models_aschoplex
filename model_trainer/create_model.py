#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import sys
import uuid

import matplotlib.pyplot as plt
import numpy as np
from dafne_dl.common.DataGenerators import DataGeneratorMem
from dafne_dl.common.preprocess_train import common_input_process_single, input_creation_mem, weighted_loss
from dafne_dl.labels.utils import invert_dict
from tensorflow.keras import optimizers
from scipy.ndimage import zoom
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback

VALIDATION_SPLIT = 0.2
BATCH_SIZE = 5
MAX_EPOCHS = 150

# Get the path of the script's parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to sys.path to import common
sys.path.append(parent_dir)

def load_data(data_path):
    """
    Load all the npz files in the folder
    :param data_path: string containing the path to the folder
    :return: list of numpy file objects
    """
    # Load all the npz files in the folder
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')]
    data_list = []
    for file in data_files:
        print(f'Loading {file}')
        npz_file = np.load(file)
        data_list.append(npz_file)
    return data_list

def sanitize_label(label):
    if label.endswith('_L'):
        label = label[:-2] + '_X'
    elif label.endswith('_R'):
        label = label[:-2] + '_Y'
    return label

def get_model_info(data_list):
    """
    Get the common resolution, size and labels from the data
    :param data_list: the list of numpy file objects
    :return: common_resolution, model_size, label_dict
    """
    # Find the most common resolution among all the data
    resolutions = [data['resolution'][:2] for data in data_list]
    common_resolution = np.median(resolutions, axis=0)[:2]

    # Find the most common size in voxel of all the slices
    sizes = [data['data'].shape[:2] for data in data_list]
    sizes_dict = {}
    for size in sizes:
        if size in sizes_dict:
            sizes_dict[size] += 1
        else:
            sizes_dict[size] = 1
    common_size = max(sizes_dict, key=sizes_dict.get)
    max_size = max(common_size)
    model_size = (max_size, max_size)

    # Create a set with all the different labels present in the files
    labels = set()
    for data in data_list:
        for key in data.keys():
            if key.startswith('mask_'):
                label = key[5:]
                labels.add(sanitize_label(label))

    labels = sorted(labels)

    # Crete a dictionary with indices for each label
    label_dict = {i+1: label for i, label in enumerate(labels)}
    return common_resolution, model_size, label_dict


def escape_string(string):
    """
    Escape a string to be used in a python source code
    :param string: the string to escape
    :return: the escaped string
    """
    return string.replace('"', '\\"')


def get_create_model_function(source):
    """
    Extract the create_model function from the source code
    :param source: source in string form
    :return: a function returning a keras model
    """
    # remove the generate_convert import from the source code
    source = source.replace('from common import generate_convert', '')

    # replace the generate_convert function with a dummy function
    def dummy_generate_convert(*args, **kwargs):
        pass

    new_locals = {'generate_convert': dummy_generate_convert}
    # now when we execute the file, only the functions will be created. This is used to extract the make_unet function
    exec(source, globals(), new_locals)
    return new_locals['make_unet']


def convert_3d_mask_to_slices(mask_dictionary):
    """
    Convert a dictionary of 3d masks to list of dictionaries of 2d masks
    :param mask_dictionary: dictionary in the form {label: mask_3d}
    :return: a list of dictionaries in the form {label: mask_2d}
    """
    mask_list = []
    for i in range(mask_dictionary[list(mask_dictionary.keys())[0]].shape[2]):
        mask_list.append({sanitize_label(label): mask[:, :, i] for label, mask in mask_dictionary.items()})
    return mask_list


def normalize_training_data(data_list, common_resolution, model_size, label_dict):
    """
    Normalize the training data
    :param data_list: the list of numpy file objects
    :param common_resolution: the common resolution of the data
    :param model_size: the common size of the data
    :param label_dict: the dictionary with the labels
    :return: list of normalized data, list of normalized masks
    """

    inverse_label_dict = invert_dict(label_dict)

    all_slice_list = []
    all_masks_list = []

    for data in data_list:
        img_3d = data['data']
        image_list = [img_3d[:, :, i] for i in range(img_3d.shape[2])]
        training_data_dict = {'image_list': image_list, 'resolution': data['resolution'][:2]}
        mask_dictionary = {key[5:]: data[key] for key in data.keys() if key.startswith('mask_')}
        mask_list = convert_3d_mask_to_slices(mask_dictionary)

        processed_image_list, processed_mask_list = common_input_process_single(inverse_label_dict,
                                                                                common_resolution,
                                                                                model_size,
                                                                                model_size,
                                                                                training_data_dict,
                                                                                mask_list,
                                                                                False)
        all_slice_list.extend(processed_image_list)
        all_masks_list.extend(processed_mask_list)

    return all_slice_list, all_masks_list


def generate_training_and_weights(data_list, mask_list, band=49):
    return input_creation_mem(data_list, mask_list, band=band)


def make_validation_list(data_list, common_resolution, model_size, label_dict):
    """
    Create a validation list
    :param data_list: the list of datasets
    :param common_resolution: resolution of the model
    :param model_size: size of the model
    :param label_dict: dictionary of the labels
    :return:
    """
    if os.path.exists('validation_obj.pickle'):
        with open('validation_obj.pickle', 'rb') as f:
            training_objects = pickle.load(f)
    else:
        normalized_data_list, normalized_mask_list = normalize_training_data(data_list,
                                                                             common_resolution,
                                                                             model_size, label_dict)
        training_objects = generate_training_and_weights(normalized_data_list, normalized_mask_list)
        with open('validation_obj.pickle', 'wb') as f:
            pickle.dump(training_objects, f)
    x_list = [np.stack([training_object[:,:,0], training_object[:,:,-1]], axis=-1) for training_object in training_objects]
    y_list = [training_object[:,:,1:-1] for training_object in training_objects]
    #plt.imshow(x_list[0][:,:,0])
    #plt.figure()
    #plt.imshow(y_list[0][:,:,2])
    #plt.figure()
    #plt.imshow(x_list[0][:,:,1])
    #plt.show()
    return x_list, y_list


def make_data_generator(data_list, common_resolution, model_size, label_dict):
    """
    Create a data generator
    :param data_list: the list of datasets
    :param common_resolution: resolution of the model
    :param model_size: size of the model
    :param label_dict: dictionary of the labels
    :return:
    """
    if os.path.exists('training_obj.pickle'):
        with open('training_obj.pickle', 'rb') as f:
            training_objects = pickle.load(f)
    else:
        normalized_data_list, normalized_mask_list = normalize_training_data(data_list,
                                                                             common_resolution,
                                                                             model_size, label_dict)
        training_objects = generate_training_and_weights(normalized_data_list, normalized_mask_list)
        with open('training_obj.pickle', 'wb') as f:
            pickle.dump(training_objects, f)
    steps = int(len(training_objects) / BATCH_SIZE)
    data_generator = DataGeneratorMem(training_objects, list_X=list(range(steps * BATCH_SIZE)),
                                               batch_size=BATCH_SIZE, dim=model_size)

    return data_generator, steps

def train_model(model, data_list, common_resolution, model_size, label_dict):
    """
    Train the model
    :param model: Keras model
    :param data_list: list of data
    :param common_resolution: resolution of the model
    :param model_size: size of the model
    :param label_dict: dictionary with labels
    :return: the trained model
    """
    n_datasets = len(data_list)
    n_validation = int(n_datasets * VALIDATION_SPLIT)

    if n_validation == 0:
        print("WARNING: No validation data will be used")

    validation_data_list = data_list[:n_validation]
    training_data_list = data_list[n_validation:]

    training_generator, steps = make_data_generator(training_data_list, common_resolution, model_size, label_dict)

    if n_validation > 0:
        x_val_list, y_val_list = make_validation_list(validation_data_list, common_resolution, model_size, label_dict)

    # now train the model on the data
    adamlr = optimizers.Adam(learning_rate=0.009765, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    model.compile(loss=weighted_loss, optimizer=adamlr)

    # do the training
    if n_validation > 0:

        plt.ion()
        class PredictionCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                segmentation = self.model.predict(np.expand_dims(x_val_list[0],0))
                #plt.imshow(x_val_list[0][:,:,0])
                #plt.figure()
                label = np.argmax(np.squeeze(segmentation[0, :, :, :-1]), axis=2)
                plt.imshow(label)
                plt.show(block=False)
                plt.pause(0.001)

        prediction_callback = PredictionCallback()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.000005)
        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor the validation loss
            mode='min',  # Stop when the monitored quantity stops decreasing
            patience=5,  # Stop if the monitored quantity does not improve after 5 epochs
            verbose=1,  # Print a message when the training is stopped
        )
        history = model.fit(training_generator, epochs=MAX_EPOCHS,
                  steps_per_epoch=steps,
                  validation_data=(np.stack(x_val_list,0), np.stack(y_val_list,0)),
                  callbacks=[prediction_callback],
                  verbose=1)
    else:
        history = model.fit(training_generator, epochs=MAX_EPOCHS, steps_per_epoch=steps, verbose=1)

    return model, history
def create_model(model_name, data_path):
    data_list = load_data(data_path)

    common_resolution, model_size, label_dict = get_model_info(data_list)

    # load the model template
    with open('generate_model.py.tmpl', 'r') as f:
        source = f.read()

    # replace the variables
    source = source.replace('%%MODEL_NAME%%', f'"{escape_string(model_name)}"')
    source = source.replace('%%MODEL_RESOLUTION%%', f'[{common_resolution[0]:.2f}, {common_resolution[1]:.2f}]')
    source = source.replace('%%MODEL_SIZE%%', f'[{model_size[0]:d}, {model_size[1]:d}]')
    source = source.replace('%%MODEL_UID%%', f'"{str(uuid.uuid4())}"')
    source = source.replace('%%LABELS_DICT%%', str(label_dict))

    # write the new model generator script
    with open(f'generate_{model_name}_model.py', 'w') as f:
        f.write(source)

    create_model_function = get_create_model_function(source)
    model = create_model_function()

    trained_model, history = train_model(model, data_list, common_resolution, model_size, label_dict)
    return trained_model, history


def save_weights(model, model_name):
    """
    Save the weights of the model
    :param model: the model
    :param model_name: the name of the model
    :return:
    """
    os.makedirs('weights', exist_ok=True)
    model_path = os.path.join('weights', f'weights_{model_name}.hdf5')
    model.save_weights(model_path)
    print(f'Saved weights to {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to create")
    parser.add_argument("data_path", help="Path to data folder (containing the '*.npz' files)")
    args = parser.parse_args()

    model, history = create_model(args.model_name, args.data_path)
    save_weights(model, args.model_name)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



