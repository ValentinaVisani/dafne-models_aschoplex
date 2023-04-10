import os
import shutil
import sys

from dafne_dl import DynamicDLModel


def generate_convert(model_id,
                     default_weights_path,
                     model_name_prefix,
                     model_create_function,
                     model_apply_function,
                     model_learn_function):
    """
    Function that either generates a new model using the default weights, or updates an existing model, based on argv[1]

    Parameters:
        model_id: the id of the model to be generated/updated
        default_weights_path: the path to the default weights
        model_name_prefix: the prefix of the model name (e.g. 'Leg' or 'Thigh'). Used for the filename
        model_create_function: the function that creates the model
        model_apply_function: the function that applies the model
        model_learn_function: the function that performs the incremental learning of the model

    Returns:
        None
    """
    if len(sys.argv) > 1:
        # convert an existing model
        print("Converting model", sys.argv[1])
        old_model_path = sys.argv[1]
        filename = old_model_path
        old_model = DynamicDLModel.Load(open(old_model_path, 'rb'))
        shutil.move(old_model_path, old_model_path + '.bak')
        weights = old_model.get_weights()
        timestamp = old_model.timestamp_id
        model_id = old_model.model_id
    else:
        model_id = model_id
        timestamp = 1610001000
        model = model_create_function()
        model.load_weights(default_weights_path)
        weights = model.get_weights()
        filename = f'models/{model_name_prefix}_{timestamp}.model'

    modelObject = DynamicDLModel(model_id,
                                 model_create_function,
                                 model_apply_function,
                                 incremental_learn_function=model_learn_function,
                                 weights=weights,
                                 timestamp_id=timestamp
                                 )

    with open(filename, 'wb') as f:
        modelObject.dump(f)

    try:
        os.remove(filename + '.sha256')
    except FileNotFoundError:
        pass

    print('Saved', filename)