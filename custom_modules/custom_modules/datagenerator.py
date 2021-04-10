import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np

from . import imaugm


class image_generator(Sequence):
    def __init__(self, gdos, Dataset, datalen, frc, batch_size,
                 input_components, output_components,
                 sequence=False, seq_batchsize=64,
                 flip=True, augm=True, proportion=0.15,
                 use_tensorboard=False, logdir="", shape=(120, 160, 3)):

        # data augmentation parameters
        self.flip = flip
        self.augm = augm
        self.proportion = proportion

        # data information
        self.gdos = gdos
        self.Dataset = Dataset
        self.datalen = datalen
        self.sequence = sequence
        self.frc = frc
        self.shape = shape

        # components information
        self.input_components = input_components
        self.output_components = output_components
        self.flipable_components = [i for i in self.input_components +
                                    self.output_components if self.Dataset.get_component(i).flip]
        self.names2index = self.Dataset.components_names2indexes()

        # batchsize information
        self.batch_size = batch_size
        self.seq_batchsize = seq_batchsize

        # tensorboard callback
        self.use_tensorboard = use_tensorboard
        self.file_writer = tf.summary.create_file_writer(
            logdir) if self.use_tensorboard else None

    def __data_generation(self):
        batchfiles = np.random.choice(self.gdos, size=self.batch_size)
        xbatch = []
        ybatch = []
        for _ in range(len(self.Dataset.get_label_structure_name())):
            ybatch.append([])

        for path in batchfiles:
            try:
                img, annotation = self.Dataset.load_img_and_annotation(path)
                if img.shape != self.shape:
                    img = cv2.resize(img, (self.shape[1], self.shape[0]))
                xbatch.append(img)
                for i in range(len(annotation)):
                    ybatch[i].append(annotation[i])
            except:
                print(path)

        if self.augm:
            xbatch, ybatch = imaugm.generate_functions_replace(
                xbatch, ybatch,
                proportion=self.proportion,
                functions=(
                    imaugm.add_random_shadow,
                    imaugm.add_rdm_noise,
                    imaugm.rescut,
                    imaugm.inverse_color,
                    imaugm.night_effect,
                )
            )

        if self.flip:
            xflip, yflip = imaugm.generate_horizontal_flip(
                self.Dataset,
                self.names2index,
                self.flipable_components,
                xbatch,
                ybatch,
                proportion=1)

            xbatch = np.concatenate((xbatch, xflip))
            ybatch = np.concatenate((ybatch, yflip), axis=1)

        xbatch = (np.array(xbatch)/255).astype(np.float32)
        ybatch = np.array(ybatch)

        # removed the weight, useless ; weight = imaugm.get_weight(ybatch, self.frc, False, acc=self.weight_acc)

        X = [xbatch]
        for i in self.input_components:
            X.append(np.float32([np.float32(tmp_array)
                                 for tmp_array in ybatch[i]]))

        Y = []
        for i in self.output_components:
            Y.append(np.float32([np.float32(tmp_array)
                                 for tmp_array in ybatch[i]]))

        # W = []
        # if 'speed' in self.names2index.keys():
        #     index = self.names2index['speed']
        #     W.append(ybatch[index])

        if self.use_tensorboard:
            with self.file_writer.as_default():
                tf.summary.image(
                    "Training data", X[0], step=0, max_outputs=self.batch_size)

        return X, Y

    def __len__(self):
        return int(self.datalen//self.batch_size)

    def __getitem__(self, index):
        return self.__data_generation()
