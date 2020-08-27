import collections
import time

import numpy as np
import tensorflow
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

from .. import architectures, imaugm
from ..datagenerator import image_generator
from ..vis import plot

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)

class LaneDetectionTrainer():
    def __init__(self, name, dataset, dospath='', dosdir=True, proportion=0.15, sequence=False,
                 smoothing=0, label_rdm=0, input_components=[], output_components=[0]):
        """Init the trainer parameters.

        Args:
            name (str): the name you want your model to be called
            dospath (str, optional): path to the directory where the images are stored. Defaults to ''.
            dosdir (bool, optional): is the dospath a directory of directory ? Defaults to True.
            proportion (float, optional): proportion of augmented images for every augmentation functions. Defaults to 0.15.
            sequence (bool, optional): wether to process the image in sequence, not supported for training yet. Defaults to False.
            smoothing (int, optional): value for label smoothing. Defaults to 0.
            label_rdm (int, optional): value for label randomization. Defaults to 0.
        """
        # self.Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])
        self.Dataset = dataset
        self.name = name
        self.dospath = dospath
        self.dosdir = dosdir
        self.sequence = sequence

        self.img_cols = 160
        self.img_rows = 120
        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.number_class = 5
        self.proportion = proportion
        self.smoothing = smoothing
        self.label_rdm = label_rdm
        self.input_components = input_components
        self.output_components = output_components

        self.callbacks = []
        self.model = None

    def build_model(self, load=False, prune=0, drop_rate=0.15, regularizer=(0.0, 0.0),
                    optimizer=tensorflow.keras.optimizers.Adam, lr=0.001,
                    loss=architectures.dir_loss, metrics=["mse"]):
        """Load a model using a model architectures from architectures.py."""
        return 
