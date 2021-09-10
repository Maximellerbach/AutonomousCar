import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tqdm import tqdm

from custom_modules.datasets import dataset_json

physical_devices = tensorflow.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


def get_latent(self, model, path):
    img, annotation = self.load_img_and_annotation(path)
    return model.predict(np.expand_dims(img / 255, axis=0))[0]


def get_latents(self, model, paths):
    latents = []
    for path in tqdm(paths):
        latents.append(get_latent(self, model, path))
    return latents


def find_nearest(a, index, th=0.1):
    idxs = []
    tmp_a = list(a)
    lat = tmp_a[index]
    del tmp_a[index]
    mse = np.mean(np.abs(tmp_a - lat), axis=-1)

    for it, loss in enumerate(mse):
        if loss < th:
            if it > index:
                it += 1
            idxs.append(it)

    return idxs


if __name__ == "__main__":
    import os

    base_path = os.path.expanduser("~") + "\\random_data"
    current_file = os.path.abspath(os.getcwd())

    model = load_model(os.path.normpath(f"{current_file}..\\test_model\\models\\fe.h5"))

    Dataset = dataset_json.Dataset(["time"])
    dos = f"{base_path}\\json_dataset\\20 checkpoint patch\\"
    paths = Dataset.load_dos_sorted(dos)

    initial_len = len(paths)
    latents = get_latents(Dataset, model, paths)

    index = 0
    del_threshold = 0.1
    while index < initial_len:
        nearests = find_nearest(latents, index=index)

        if len(nearests) > 0:
            nearests.sort()
            nearests = list(reversed(nearests))

            for near in nearests:
                # gets nearest images and displays it
                img, annotation = Dataset.load_img_and_annotation(paths[near])
                cv2.imshow("img", img)
                cv2.waitKey(0)

        index += 1
