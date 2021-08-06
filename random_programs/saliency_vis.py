import os

import cv2
import numpy as np
import tensorflow
from custom_modules import architectures
from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_fe, vis_lab

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)

base_path = os.path.expanduser("~") + "\\random_data"
dos = f'{base_path}\\test_scene\\'
Dataset = dataset_json.Dataset(["direction", "speed", "throttle"])
input_components = []

gdos = Dataset.load_dataset_sorted(dos, flat=True)
np.random.shuffle(gdos)

model = architectures.safe_load_model(
    'test_model\\models\\auto_label5.h5', compile=False)
architectures.apply_predict_decorator(model)
model.summary()

vis_model = tensorflow.keras.models.Model(model.inputs, model.layers[-1].output)

for labpath in gdos:
    img, annotation = Dataset.load_img_and_annotation(labpath, to_list=False)

    to_pred = Dataset.make_to_pred_annotations([img], [annotation], input_components)
    prediction_dict, dt = model.predict(to_pred)
    img = img/255

    for class_idx in range(vis_model.output.shape[1]):
        vis_img = vis_fe.get_gradcam(
            vis_model, to_pred, class_idx, penultimate_layer=2)
        vis_img = np.transpose(vis_img[0], (1, 2, 0))

        # cv2.imshow(f'img_{class_idx}', vis_img * img)
        cv2.imshow(f'{class_idx}', vis_img)

    vis_lab.vis_all_compare(Dataset, input_components,
                            img, annotation, prediction_dict[0], waitkey=None)
    cv2.waitKey(0)
