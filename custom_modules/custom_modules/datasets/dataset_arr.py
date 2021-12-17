from glob import glob
import numpy as np
import cv2


class Dataset:
    """Dataset class that contains everything needed to load and save a json/arr dataset."""

    def __init__(self):
        """Init the class.

        Args:
            lab_structure (list): list of components (class)
        """
        self.__meta_components = []
        self.data_structure = ["direction", "speed", "throttle", "time"]
        self.format = ".npy"

    def get_memmap(self, path: str, *args, **kwargs):
        memmap = np.memmap(path, *args, **kwargs)
        return memmap

    def get_image_and_memmap(self, path: str, *args, **kwargs):
        memmap = np.memmap(path, *args, **kwargs)
        image = cv2.imread(path.split(".")[-1]+".png")

        return image, memmap

    def get_annotation_dict(self, path: str):
        arr = np.load(path)
        annotation = dict(zip(self.data_structure, arr))
        return annotation

    def save_annotation_arr(self, arr, path: str):
        np.save(path, arr)

    def save_annotation_dict(self, annotation: dict, path: str):
        arr = np.array(annotation.values())
        self.save_annotation_arr(arr, path)


if __name__ == "__main__":
    Dataset = Dataset()

    arr = np.array([0.5, 12, 0.63, 123982138912313.123])
    Dataset.save_annotation_arr(arr, "thisisatest.npy")

    memmap = Dataset.get_memmap("thisisatest.npy", mode='r', shape=(4,))
    print(memmap)

    annotation = Dataset.get_annotation_dict("thisisatest.npy")
    print(annotation)
