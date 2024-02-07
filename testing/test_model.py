#!/bin/env python3
# -*- coding: utf-8 -*-
# Example usage:
# testing/test_model.py --classification 'right' generate_leg_split_model.py ~/.local/share/Dafne/models/Leg_1669385545.model ../../dicom/Dafne_Erlangen/1663761356.npz 12

import argparse

import numpy as np
from dafne_dl import DynamicDLModel

def generate_convert_replacement(model_id,
                     default_weights_path,
                     model_name_prefix,
                     model_create_function,
                     model_apply_function,
                     model_learn_function):
    global new_generated_model
    import time
    new_generated_model = DynamicDLModel(model_id,
                                 model_create_function,
                                 model_apply_function,
                                 incremental_learn_function=model_learn_function,
                                 timestamp_id=int(time.time())
                                 )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", help="Path to model source code")
    parser.add_argument("model_path", help="Path to model file containing the weights to test")
    parser.add_argument("test_path", help="Path to test data")
    parser.add_argument("slice", type=int, help="Slice to show")
    parser.add_argument("--classification", nargs=1, metavar='class', type=str, default='', required=False, help="Optional classification label")
    args = parser.parse_args()

    if args.classification:
        classification = args.classification[0]
    else:
        classification = ''

    # open the source file
    with open(args.source_path, 'r') as f:
        source = f.read()

    # replace the generate_convert function
    new_locals = {'generate_convert': generate_convert_replacement}
    # thanks to the new generate_convert function, the model will be generated in the new_generated_model global variable
    exec(source, globals(), new_locals)

    # load the model
    with open(args.model_path, 'rb') as f:
        original_model = DynamicDLModel.Load(f)

    # set the weights to the new model
    new_generated_model.set_weights(original_model.get_weights())

    # load the test data
    with np.load(args.test_path) as f:
        data = f['data'].astype(np.float32)
        resolution = f['resolution'][:2]

    print(resolution)

    slice_to_display = data[:, :, args.slice]
    # apply the model
    output = new_generated_model.apply({'image': slice_to_display,
                                        'resolution': resolution,
                                        'split_laterality': False,
                                        'classification': classification})

    # create a single mask

    accumulated_mask = np.zeros_like(slice_to_display, dtype=np.int32)
    for val, mask in enumerate(output.values()):
        accumulated_mask += mask*(val+1)

    # display the slice and the masks
    import matplotlib.pyplot as plt
    plt.imshow(slice_to_display, cmap='gray')
    plt.imshow(accumulated_mask, alpha=0.5, cmap='jet')
    plt.show()

if __name__ == '__main__':
    main()