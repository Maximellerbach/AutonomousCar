import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

import pred_function
from customDataset import DatasetJson


Dataset = DatasetJson(["direction", "speed", "throttle", "time"])
doss = "C:\\Users\\maxim\\random_data\\json_dataset\\"

model = load_model('test_model\\models\\linear_trackmania2.h5',
                   compile=False)
fe = load_model('test_model\\models\\fe.h5')

layers_info = [(it, layer.name)
               for it, layer in enumerate(fe.layers)]


gdos = Dataset.load_dataset(doss)
gdos = np.concatenate([i for i in gdos])
np.random.shuffle(gdos)

filter_indexes = []
for layer in layers_info:
    if 'activation' in layer[1]:
        filter_indexes.append(layer[0])

for labpath in gdos:
    img, annotations = Dataset.load_img_and_annotation(labpath)
    pred = model.predict(np.expand_dims(img/255, axis=0))[0][0]
    print(f'steering: {pred}')

    cv2.imshow('img', img/255)
    for index in filter_indexes[:4]:  # only get the first 4
        activations = pred_function.visualize_model_layer_filter(
            fe, img, index, show=False)

    cv2.waitKey(1)
    plt.show()
