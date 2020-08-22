import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model

from custom_modules import architectures, pred_function
from customDataset import DatasetJson


Dataset = DatasetJson(["direction", "speed", "throttle", "time"])
doss = "C:\\Users\\maxim\\random_data\\json_dataset\\"

with tfmot.sparsity.keras.prune_scope():
    model = load_model('test_model\\models\\linear_trackmania.h5',
                       compile=False)
fe = architectures.get_fe(model)

filter_indexes = []
for it, layer in enumerate(fe.layers):
    if 'activation' in layer.name:
        filter_indexes.append(it)

gdos = Dataset.load_dataset(doss)
gdos = np.concatenate([i for i in gdos])
np.random.shuffle(gdos)

for labpath in gdos:
    img, annotations = Dataset.load_img_and_annotation(labpath)
    pred = model.predict(np.expand_dims(img/255, axis=0))[0][0]
    print(f'steering: {pred}')

    cv2.imshow('img', img/255)
    for index in filter_indexes[:4]:  # only get the first 4
        activations = pred_function.visualize_model_layer_filter(
            fe, img, index, input_size=(80, 60), show=False)

    cv2.waitKey(0)
    plt.show()
