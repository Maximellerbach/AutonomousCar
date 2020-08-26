import cv2
import numpy as np
import tensorflow

from custom_modules import architectures
from custom_modules.datasets import dataset_json

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


class LabelisationSuggestion():
    def __init__(self, Dataset, model_path):
        self.Dataset = Dataset

        model = architectures.safe_load_model(model_path, compile=False)
        self.fe = architectures.get_flat_fe(model)

    def compute_distance(self, a, b):
        return np.linalg.norm(a-b)

    def load_imgs(self, paths, flip=False):
        X = []
        for i in paths:
            img = self.Dataset.load_img(i)/255
            if flip:
                imgflip = cv2.flip(img, 1)
                X.append(img)
                X.append(imgflip)
            else:
                X.append(img)
        return np.array(X)

    def load_paths(self, dos):
        return self.Dataset.load_dos_sorted(dos)

    def get_pred(self, X):
        return self.fe.predict(X)

    def compare_lat(self, lat_to_compare, lat_imgs, N=5):
        distances = [self.compute_distance(
            lat_to_compare, lat_img) for lat_img in lat_imgs]
        sorted_distances = sorted(distances)
        smallest = sorted_distances[:N]

        return distances, smallest

    def load_annotations_from_indexes(self, paths, indexes):
        return [self.Dataset.load_annotation(paths[i]) for i in indexes]

    def estimate_annotation(self, annotations, distances):
        def softmax(z):
            s = np.max(z, axis=1)
            s = s[:, np.newaxis]  # necessary step to do broadcasting
            e_x = np.exp(z - s)
            div = np.sum(e_x, axis=1)
            div = div[:, np.newaxis]  # dito
            return e_x / div

        av_weights = 1-softmax([distances])[0]
        return list(np.average(annotations, weights=av_weights, axis=0))

    def analyze(self, dos):
        paths = lab_helper.load_paths(dos)
        imgs = self.load_imgs(paths)
        lat_imgs = self.get_pred(imgs)

        for i in range(len(lat_imgs)):
            # don't analyse previous images
            distances, smallest_distance = self.compare_lat(
                lat_imgs[i], lat_imgs[i+1:])
            indexes = []
            for dist in smallest_distance:
                indexes.append(distances.index(dist)+i)

            annotations = self.load_annotations_from_indexes(paths, indexes)
            predicted = self.estimate_annotation(
                annotations, smallest_distance)
            annotation = self.Dataset.load_annotation(paths[i])
            # print(annotation, predicted)

            print(list(np.array(annotation)-np.array(predicted)))
        return


if __name__ == "__main__":
    Dataset = dataset_json.Dataset(['direction', 'speed', 'throttle', 'time'])
    lab_helper = LabelisationSuggestion(
        Dataset, 'test_model\\models\\linear_trackmania.h5')

    lab_helper.analyze(
        "C:\\Users\\maxim\\random_data\\json_dataset\\20 checkpoint patch\\")
