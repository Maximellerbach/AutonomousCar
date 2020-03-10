import json
from glob import glob

import keras
import keras.backend as K
from keras.models import load_model

def dir_loss(y_true, y_pred):
    return K.sqrt(K.square(y_true-y_pred))


def get_models_path(path='test_model\\convolution\\*.h5'):
    return glob(path)

def layers_to_dict(model):
    mod_dict = []
    for i in model.layers:
        mod_dict.append(i.get_config())
    return mod_dict

if __name__ == "__main__":
    paths = get_models_path()
    for path in paths:
        model = load_model(path, custom_objects={"dir_loss":dir_loss})
        infos = layers_to_dict(model)
        model_name = path.split('\\')[-1].split('.h5')[0]
        with open('test_model\\convolution\\models_json\\'+model_name+'.json', 'w') as f:
            json.dump(infos, f)
