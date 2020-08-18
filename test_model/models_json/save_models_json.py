import json
from glob import glob

import keras
import keras.backend as K
from keras.models import load_model
from keras.utils import plot_model

def dir_loss(y_true, y_pred):
    return K.sqrt(K.square(y_true-y_pred))


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

        if 'test_model\\models\\models_json\\'+model_name+'.json' not in glob('test_model\\models\\models_json\\*.json'):
            if model == None:
                model = load_model(path, custom_objects={"dir_loss":dir_loss})

            with open('test_model\\models\\models_json\\'+model_name+'.json', 'w') as f:
                infos = layers_to_dict(model)
                json.dump(infos, f)

        if 'test_model\\models\\models_json\\'+model_name+'.png' not in glob('test_model\\models\\models_json\\*.png'):
            if model == None:
                model = load_model(path, custom_objects={"dir_loss":dir_loss})
            plot_model(model, to_file='test_model\\models\\models_json\\'+model_name+'.png')
