#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from dafne_models.ui.ModelTrainer import main

if __name__ == '__main__':
    main()
