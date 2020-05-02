import cv2
import numpy as np
from keras.models import load_model
from tqdm import tqdm

from data_class import Data

def compute_distance(a, b):
    return np.linalg.norm(a-b)

def get_latent(self, model, path):
    img = self.load_img(path)/255
    latent = model.predict(np.expand_dims(img, axis=0))
    return

def get_latents(self, model, paths):
    latents = []
    for path in tqdm(paths):
        latents.append(get_latent(self, model, path))
    return latents

if __name__ == "__main__":
    model = load_model("test_model\\convolution\\fe.h5")
    data = Data("C:\\Users\\maxim\\datasets\\1 ironcar driving\\", is_float=False, recursive=False)

    latents = get_latents(data, model, data.dts)

