#!/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import json

from visualize_groundtruth import create_segmentation_example

WEBSITE_MODEL_PATH = '/models'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to the json model description")
    parser.add_argument("bundle_file", help="Path to the npz bundle")
    parser.add_argument("output_path", help="Path to the output folder")
    parser.add_argument("--slice", type=int, default=None, required=False, help="Slice to show")

    args = parser.parse_args()
    bundle_file = args.bundle_file
    slice = args.slice
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    create_segmentation_example(bundle_file, os.path.join(output_folder, 'segmentation.png'), slice)

    while output_folder.endswith('/'):
        output_folder = output_folder[:-1]

    output_file_content = \
f"""---
title: {os.path.basename(output_folder)}
image: {WEBSITE_MODEL_PATH}/{os.path.basename(output_folder)}/segmentation.png
layout: page
---
## Info
"""

    # process JSON file
    with open(args.json_file, 'r') as f:
        model_info = json.load(f)

    for key, value in model_info['info'].items():
        output_file_content += f"*  **{key}**: {value}\n"

    output_file_content += '\n\n'
    variants = model_info.get('variants', [])
    if variants:
        output_file_content += '## Variants\n'
        for variant in variants:
            if variant == '':
                output_file_content += "*  *(base)*\n"
            else:
                output_file_content += f"*  {variant}\n"

    output_file_content += '\n\n## Categories\n'
    for category in model_info['categories']:
        output_file_content += f"*  {' -> '.join(category)}\n"

    with open(os.path.join(output_folder, 'index.md'), 'w') as f:
        f.write(output_file_content)