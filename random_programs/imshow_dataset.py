import cv2
import numpy as np

import tensorflow
from custom_modules import architectures, pred_function
from custom_modules.vis import vis_fe
from custom_modules.datasets import dataset_json

import os
base_path = os.getenv('ONEDRIVE') + "\\random_data"

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = [1]

dos = f'{base_path}\\recorded_imgs\\0_1600003784.089919\\'
# dos = 'C:\\Users\\maxim\\recorded_imgs\\0_1600794409.9635623\\'

model = architectures.safe_load_model(
    'test_model\\models\\rbrl_sim4.h5', compile=False)
architectures.apply_predict_decorator(model)

fe = architectures.get_fe(model)
fe.summary()

filter_indexes = []
for it, layer in enumerate(fe.layers):
    if 'activation' in layer.name:
        filter_indexes.append(it)

gdos = Dataset.load_dos_sorted(dos)
np.random.shuffle(gdos)

for labpath in gdos:
    img, annotation = Dataset.load_img_and_annotation(labpath)
    pred_function.test_predict_paths(
        Dataset, input_components, model, [labpath], waitkey=None, apply_decorator=False)

    img = img/255
    for index in filter_indexes[:5]:
        activations, _ = vis_fe.visualize_model_layer_filter(
            fe, img, index, output_size=(80, 60), show=False)

    cv2.waitKey(0)
