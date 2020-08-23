import json
from glob import glob

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def get_models_path(path='test_model\\models\\*.h5'):
    return glob(path)

def layers_to_dict(model):
    mod_dict = []
    for i in model.layers:
        mod_dict.append(i.get_config())
    return mod_dict

if __name__ == "__main__":
    paths = get_models_path()
    for path in paths:
        model = None
        model_name = path.split('\\')[-1].split('.h5')[0]
        base_path = f'test_model\\models\\models_json\\{model_name}'

        if base_path+'.json' not in glob('test_model\\models\\models_json\\*.json'):
            if model is None:
                model = load_model(path, compile=False)

            with open(base_path+'.json', 'w') as json_file:
                infos = layers_to_dict(model)
                json.dump(infos, json_file)

        if base_path+'.png' not in glob('test_model\\models\\models_json\\*.png'):
            if model is None:
                model = load_model(path, compile=False)
            plot_model(model, to_file=base_path+'.png')
