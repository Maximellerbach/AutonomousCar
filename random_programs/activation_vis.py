import os

import cv2
import numpy as np
import tensorflow
from custom_modules import architectures
from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_fe, vis_lab

base_path = os.path.expanduser("~") + "\\random_data"
dos = f"{base_path}\\donkeycar\\"

physical_devices = tensorflow.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


Dataset = dataset_json.Dataset(["direction", "speed", "throttle"])
input_components = []

gdos = Dataset.load_dataset(dos, flat=True)
np.random.shuffle(gdos)

model = architectures.safe_load_model("test_model\\models\\auto_label5.h5", compile=False)
architectures.apply_predict_decorator(model)

fe = architectures.get_fe(model)
fe.summary()

filter_indexes = []
for it, layer in enumerate(fe.layers):
    if "activation" in layer.name:
        filter_indexes.append(it)


for labpath in gdos:
    img, annotation = Dataset.load_img_and_annotation(labpath, to_list=False)

    for index in filter_indexes:
        activations, _ = vis_fe.visualize_model_layer_filter(fe, img / 255, index, output_size=(80, 60), show=True)

    to_pred = Dataset.make_to_pred_annotations([img], [annotation], input_components)
    prediction_dict, dt = model.predict(to_pred)
    img = vis_lab.vis_all_compare(Dataset, input_components, img, annotation, prediction_dict[0], show=False)

    cv2.imshow("image", img)
    cv2.waitKey(0)
