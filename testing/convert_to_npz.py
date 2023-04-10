#!/bin/env python3
# -*- coding: utf-8 -*-

from dafne.utils.dicomUtils.misc import dosma_volume_from_path
import numpy as np
import sys
import os

def main():
    path = sys.argv[1]

    med_volume, *_ = dosma_volume_from_path(path)
    if os.path.isdir(path):
        out_path = os.path.join(path, 'data.npz')
    else:
        out_path = path + '.npz'
    np.savez(out_path, data=med_volume.volume, resolution=med_volume.pixel_spacing)

if __name__ == '__main__':
    main()