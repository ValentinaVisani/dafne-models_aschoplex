# dafne-models
Repository for model generators

## Important!
git-lfs is required to properly check out the model weights for the "thigh" and "leg" models.

## How to use the model trainer
The model trainer is a python script that can be used to train a model on a set of images.
It is based on the keras library and uses a tensorflow backend. The model architecture is 2D and is the 
same as the "thigh" and "leg" models, that is, a modified V-Net.

The input data is in the form of "Numpy bundle" files, which contain the image dataset with the "data" key,
the resolution under the "resolution" key, in the form of a numpy vector with 2 or 3 elements corresponding
to the voxel size in the three dimensions, and various binary masks in the form of 3D numpy arrays with the same
dimensions as the data with "mask_<roi1>", "mask_<roi2>" etc. keys (where <roi1> and <roi2> are the names of the
regions of interest, e.g. mask_tibia, mask_fibula, etc).

This numpy bundle can (and should) be saved from Dafne by selecting the "Export masks"->"Numpy bundle" option.

Once multiple such datasets are collected under the same folder, the model trainer can be used.

### Usage
The recommended usage is by using the GUI for fitting, as it allows a wider range of options.
Assuming you have cloned the dafne_models repository, install it locally together with its dependencies with

```pip install -e .[gui]```

Then, run the GUI with

```python create_model_ui.py```

or simply with

```create_model_ui```

Load the data by clicking on "Choose" next to the "Data location" field and selecting the folder containing 
the numpy bundles.
Then, select the model name and the output folder and click on "Fit model". The model will start training.
You can monitor the progress from the plots that will be displayed. On the right, one segmentation of a validation
slice is displayed at the end of every epoch. The slider on the bottom chooses which validation slice is displayed.
Next to it, a checkbox lets the user choose whether the training should be stopped when the validation loss starts
to increase. This is useful to avoid overfitting, but the user can choose to stop it manually instead.

The model will be saved in the output folder as a keras model (.hdf5 file) and as a dafne model (.model file). A .py
file is also saved to build the .model file from the .hdf5 file.

### Importing a new model into Dafne
The model can be imported into Dafne by selecting "Local" as the model location in the settings, and then choosing
"Import model" from the "File" menu. If ou don't see "Import model" under the "File" menu, double-check that the
Model location is set to "Local".

### Caveats
It is highly recommended to have a correctly configured GPU for the training. Make sure that you have tensorflow
installed for the python version that you are using, and that the GPU libraries are compatible with this tensorflow
version.

## Command line usage
A command-line model trainer is also provided. It can be used to train a model from the command line, without
the GUI. It is recommended to use the GUI instead, as it allows a wider range of options.

To train a model from the command line, run

```python create_model.py <model_name> <data folder>```

where <data_folder> is the folder containing the numpy bundles, and <model_name> is the name of the model
to be trained. The model will be saved in the current folder.