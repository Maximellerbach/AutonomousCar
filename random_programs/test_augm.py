import os

import cv2
import numpy as np
from custom_modules import imaugm, architectures
from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_lab


def apply_augm(impath, n=1, proportion=0.1):
    img, annotation = Dataset.load_img_and_annotation(impath, to_list=True)
    dict_annotation = Dataset.load_annotation(impath, to_list=False)
    print(annotation)
    xbatch = np.array([img for _ in range(n)])
    ybatch = np.array([annotation for _ in range(n)])
    ybatch = np.expand_dims(ybatch, 0)

    # print(xbatch.shape)
    # print(ybatch.shape)

    xbatch, ybatch = imaugm.generate_functions_replace(
        xbatch,
        ybatch,
        proportion=proportion,
        functions=(
            imaugm.add_blur,
            imaugm.add_random_shadow,
            imaugm.add_rdm_noise,
            imaugm.rescut,
            # imaugm.change_brightness,
            # imaugm.inverse_color,
            # imaugm.night_effect,
        ),
    )

    for i, img in enumerate(xbatch):
        yield (img, dict_annotation, i)


if __name__ == '__main__':
    model = architectures.safe_load_model(
        "test_model\\models\\test_renault.tflite", output_names=["direction", "throttle"])

    base_path = os.path.expanduser("~") + "\\random_data"
    dos = f"{base_path}\\donkeycar\\20-11-21\\1\\"

    Dataset = dataset_json.Dataset(["direction", "speed", "throttle"])
    gdos = Dataset.load_dos_sorted(dos)

    for path in gdos:
        for (img, annotation, i) in apply_augm(path):
            to_pred = Dataset.make_to_pred_annotations([img], [annotation], [])
            prediction_dict, elapsed_time = model.predict(to_pred)

            # cv2.imshow(f'img_{i}', img)
            vis_lab.vis_all_compare(Dataset, [], img, annotation, prediction_dict)
            cv2.waitKey(0)
