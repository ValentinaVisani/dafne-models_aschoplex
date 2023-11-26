#!/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

def get_colormap(max_labels):
    # Define the number of segments (including the transparent color for zero)
    num_segments = max_labels + 1

    # Create a list of colors with varying alpha values
    colors = [(0, 0, 0, 0)]  # Transparent color for zero

    cmap = mpl.colormaps['hsv']
    n_colors = cmap.N

    for i in range(1, num_segments):
        # Generate distinguishable colors using HSV color space
        color = cmap(int(i * n_colors / num_segments))
        color = (color[0], color[1], color[2], 0.5)
        colors.append(color)

    # Create the colormap
    return ListedColormap(colors)


def find_best_slice(masks, total_slices):
    # find the slices with the most masks
    mask_counts = []
    total_masks = len(masks)
    for slice in range(total_slices):
        count = 0
        for mask in masks.values():
            if np.any(mask[:, :, slice]):
                count += 1
        mask_counts.append(count)

    # find all the slices with the most masks
    max_count = max(mask_counts)
    if max_count < total_masks:
        print("Warning: not all masks are present in the image at the same time")

    best_slices = []
    for slice, count in enumerate(mask_counts):
        if count == max_count:
            best_slices.append(slice)

    # find the slice with the most masks in the middle
    middle = len(best_slices) // 2
    return best_slices[middle]




def main(bundle_path, output_path=None, slice=None):


    masks = {}

    with np.load(bundle_path) as bundle:
        data = bundle['data']
        resolution = bundle['resolution']
        for key in bundle:
            if key.startswith('mask_'):
                masks[key[5:]] = bundle[key]

    slice = find_best_slice(masks, data.shape[2]) if slice is None else slice
    print("Showing slice", slice)

    img = data[:, :, slice].astype(np.float32)
    accumulated_mask = np.zeros_like(img, dtype=np.uint8)
    mask_labels = []
    current_mask_index = 1
    for mask_name, mask in masks.items():
        accumulated_mask += mask[:, :, slice] * current_mask_index
        mask_labels.append(mask_name)
        current_mask_index += 1

    # Create a figure with two subplots, ensuring they have the same height
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

    fig.tight_layout()
    fig.set_tight_layout(True)

    colors = get_colormap(len(mask_labels))

    print(colors(1))

    ax_left.imshow(img, cmap='gray')
    ax_left.imshow(accumulated_mask, cmap=colors, interpolation='none')
    ax_left.axis('off')

    # Loop over the list of strings and colors
    for i, text in enumerate(mask_labels):
        # Add a rectangle with the corresponding color, making it as tall as the text
        rect_height = 0.3  # Height of the rectangle, roughly matching text height
        rect = Rectangle((0, i + 0.5 - rect_height / 2), 1, rect_height, color=colors(i + 1))
        ax_right.add_patch(rect)

        # Add the text next to the rectangle
        ax_right.text(1.2, i + 0.5, text, va='center')
        print(text, colors(i + 1))

    # Set the limits and turn off the axes
    ax_right.set_xlim(0, 4)
    ax_right.set_ylim(0, len(mask_labels))
    ax_right.axis('off')

    # save the figure
    if output_path is not None:
        output_path = bundle_path + '.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_path", help="Path to the npz bundle")
    parser.add_argument("--slice", type=int, default=None, required=False, help="Slice to show")

    args = parser.parse_args()
    bundle_path = args.bundle_path
    slice = args.slice
    main(bundle_path, slice)