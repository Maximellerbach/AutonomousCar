import cv2
import numpy as np

import tensorflow
from custom_modules import architectures
from custom_modules.vis import vis_fe
from custom_modules.datasets import dataset_json


physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
dos = "C:\\Users\\maxim\\random_data\\json_dataset\\1 ironcar driving\\"

model = architectures.safe_load_model(
    'test_model\\models\\test.h5', compile=False)
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

    img = img/255
    print(img.shape)
    cv2.imshow('img', img)
    for index in filter_indexes[:4]:  # only get the first 4
        activations = vis_fe.visualize_model_layer_filter(
            fe, img, index, output_size=(80, 60), show=False)

    cv2.waitKey(0)
