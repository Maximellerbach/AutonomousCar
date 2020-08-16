import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

import pred_function
from customDataset import DatasetJson


Dataset = DatasetJson(["direction", "speed", "throttle", "time"])
doss = "C:\\Users\\maxim\\random_data\\json_dataset\\"

fe = load_model("test_model\\convolution\\fe.h5")

layers_info = [(it, layer.name)
               for it, layer in enumerate(fe.layers)]

# for layer in layers_info:
#     print(layer)

gdos = Dataset.load_dataset(doss, max_interval=0.1)
gdos = np.concatenate([i for i in gdos])
np.random.shuffle(gdos)

for labpath in gdos:
    img, annotations = Dataset.load_img_and_annotation(labpath)
    indexes = [4, 7, 10, 14]
    for index in indexes:
        cv2.imshow('img', img)
        cv2.waitKey(1)
        pred_function.visualize_model_layer_filter(
            fe, img, index)
