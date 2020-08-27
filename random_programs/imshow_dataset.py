import cv2
import matplotlib.pyplot as plt
import numpy as np

from custom_modules import architectures, pred_function
from custom_modules.vis import vis_fe
from custom_modules.datasets import dataset_json


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
doss = "C:\\Users\\maxim\\random_data\\json_dataset\\"

model = architectures.safe_load_model(
    'test_model\\models\\linear_trackmania2.h5', compile=False)
fe = architectures.get_fe(model)

filter_indexes = []
for it, layer in enumerate(fe.layers):
    if 'activation' in layer.name:
        filter_indexes.append(it)

gdos = Dataset.load_dataset(doss)
gdos = np.concatenate([i for i in gdos])
np.random.shuffle(gdos)

for labpath in gdos:
    img, annotation = Dataset.load_img_and_annotation(labpath)
    pred = model.predict(np.expand_dims(img/255, axis=0))[0][0]
    print(f'steering: {pred}')

    cv2.imshow('img', img/255)
    for index in filter_indexes[:4]:  # only get the first 4
        activations = vis_fe.visualize_model_layer_filter(
            fe, img, index, output_size=(80, 60), show=False)

    cv2.waitKey(0)
    plt.show()
