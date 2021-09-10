import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn import manifold

from custom_modules import architectures
from custom_modules.datasets import dataset_json

physical_devices = tensorflow.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


class data_visualization:
    def __init__(self, Dataset, model_path):
        self.Dataset = Dataset

        model = architectures.safe_load_model(model_path, compile=False)
        self.fe = architectures.get_flat_fe(model)

    def load_imgs(self, paths, flip=False):
        X = []
        for i in paths:
            img = self.Dataset.load_img(i) / 255
            if flip:
                imgflip = cv2.flip(img, 1)
                X.append(img)
                X.append(imgflip)
            else:
                X.append(img)

        return X

    def get_batchs(self, paths, max_img=2000, scramble=True):
        if scramble:
            random.shuffle(paths)

        batchs = []
        for i in range(len(paths) // max_img):
            batchs.append(paths[i * max_img : (i + 1) * max_img])
        return batchs

    def get_pred(self, X):
        return self.fe.predict(np.array(X))

    def normalize(self, data):
        return data / np.amax(data)

    def imscatter(self, x, y, ax, imageData, zoom):
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            # Convert to image
            img = imageData[i] * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0), xycoords="data", frameon=False)
            ax.add_artist(ab)

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    def computeTSNEProjectionOfLatentSpace(self, dos, doss=False, display=True):
        if doss:
            paths = self.Dataset.load_dataset(dos, flat=True)
        else:
            paths = self.Dataset.load_dos_sorted(dos)
        batchs = self.get_batchs(paths, max_img=1000)
        print(len(batchs), len(paths))

        for batch in batchs:
            X = self.load_imgs(batch, flip=True)
            X_encoded = self.get_pred(X)
            X_encoded = self.normalize(X_encoded)

            # print("Computing t-SNE embedding...")
            tsne = manifold.TSNE(n_components=2, init="pca", random_state=0, learning_rate=200)
            X_tsne = tsne.fit_transform(X_encoded)
            print(X_tsne.shape)

            if display:
                fig, ax = plt.subplots()
                self.imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.4)
                plt.show()


if __name__ == "__main__":
    import os

    base_path = os.path.expanduser("~") + "\\random_data"

    Dataset = dataset_json.Dataset(["direction", "speed", "throttle"])

    vis = data_visualization(Dataset, "test_model\\models\\auto_label5.h5")
    vis.computeTSNEProjectionOfLatentSpace(f"{base_path}\\test_scene\\", doss=True, display=True)
    # vis.clustering(doss)
