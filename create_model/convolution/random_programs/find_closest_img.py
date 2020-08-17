import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from tqdm import tqdm

from customDataset import DatasetJson

config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# to log device placement (on which device the operation ran)
config.log_device_placement = True
set_session(sess)  # set this TensorFlow session as the default


def get_latent(self, model, path):
    img, annotations = self.load_img_and_annotation(path)
    return model.predict(np.expand_dims(img/255, axis=0))[0]


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
    mse = np.mean(np.abs(tmp_a-lat), axis=-1)

    for it, loss in enumerate(mse):
        if loss < th:
            if it > index:
                it += 1
            idxs.append(it)

    return idxs


if __name__ == "__main__":
    model = load_model(
        "C:\\Users\\maxim\\github\\AutonomousCar\\test_model\\convolution\\fe.h5")

    Dataset = DatasetJson(["time"])
    dos = "C:\\Users\\maxim\\random_data\\json_dataset\\20 checkpoint patch\\"
    paths = Dataset.load_dos_sorted(dos)

    initial_len = len(paths)
    latents = get_latents(Dataset, model, paths)

    index = 0
    del_threshold = 0.1
    while(index < initial_len):
        nearests = find_nearest(latents, index=index)

        if len(nearests) > 0:
            nearests.sort()
            nearests = list(reversed(nearests))

            for near in nearests:
                # gets nearest images and displays it
                img, annotations = Dataset.load_img_and_annotation(paths[near])
                cv2.imshow('img', img)
                cv2.waitKey(0)

        index += 1
