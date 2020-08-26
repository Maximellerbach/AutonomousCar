import time

import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np

from custom_modules import autolib


class image_generator(Sequence):
    def __init__(self, gdos, Dataset, datalen, frc,
                 input_components, output_components,
                 batch_size, sequence=False, seq_batchsize=64,
                 flip=True, augm=True, proportion=0.15,
                 use_tensorboard=False, logdir="", shape=(160, 120, 3)):

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

        # batchsize information
        self.batch_size = batch_size
        self.seq_batchsize = seq_batchsize

        # tensorboard callback
        self.use_tensorboard = use_tensorboard
        self.file_writer = tf.summary.create_file_writer(
            logdir) if self.use_tensorboard else None

    def __data_generation(self, gdos):
        batchfiles = np.random.choice(gdos, size=self.batch_size)
        xbatch = []
        ybatch = []

        map(self.__load_img_and_annotation, [
            (xbatch, ybatch, path) for path in batchfiles])

        if self.augm:
            autolib.generate_functions_replace(
                xbatch, ybatch,
                proportion=self.proportion,
                functions=(
                    autolib.change_brightness,
                    autolib.rescut,
                    autolib.inverse_color,
                    autolib.add_random_shadow,
                    autolib.add_random_glow,
                    autolib.rdm_noise
                )
            )

        if self.flip:
            xflip, yflip = autolib.generate_horizontal_flip(
                xbatch, ybatch, proportion=1)
            xbatch = self.__normalize(np.concatenate((xbatch, xflip)))
            ybatch = np.concatenate((ybatch, yflip))

        # removed the weight, useless ; weight = autolib.get_weight(ybatch, self.frc, False, acc=self.weight_acc)

        X = [xbatch]
        for i in self.input_components:
            X.append(ybatch[:, i])

        Y = []
        for i in self.output_components:
            Y.append(ybatch[:, i])

        if self.use_tensorboard:
            with self.file_writer.as_default():
                tf.summary.image(
                    "Training data", X[0, 0], step=0, max_outputs=16)
        return X, Y

    def __data_generation_seq(self, gdos):
        batchfiles = np.random.choice(gdos, size=self.batch_size)
        xbatch = []
        ybatch = []

        for i in batchfiles:
            xseq = []
            yseq = []

            seq_len = len(i)
            if seq_len > self.seq_batchsize:
                rdm_seq = np.random.randint(0, seq_len-self.seq_batchsize)
                i = i[rdm_seq:rdm_seq+self.seq_batchsize]

            elif seq_len < self.seq_batchsize:  # ugly way to make sure every sequence has the same length
                i = [i[0]]*(self.seq_batchsize-seq_len)+i

            map(self.__load_img_and_annotation, [
                (xseq, yseq, path) for path in i])

            xbatch.append(xseq)
            ybatch.append(yseq)

        if self.augm:
            for i in range(len(xbatch)):
                xbatch[i], ybatch[i] = autolib.generate_functions_replace(
                    xbatch[i], ybatch[i],
                    proportion=self.proportion,
                    functions=(
                        autolib.change_brightness,
                        autolib.rescut,
                        autolib.inverse_color,
                        autolib.add_random_shadow,
                        autolib.add_random_glow,
                        autolib.rdm_noise
                    )
                )

        if self.flip:
            for x, y in zip(xbatch, ybatch):
                xflip, yflip = autolib.generate_horizontal_flip(
                    x, y, proportion=1)
                xbatch.append(xflip)
                ybatch.append(yflip)

            xbatch = self.__normalize(xbatch)
            ybatch = np.array(ybatch)

        X = [xbatch]
        for i in self.input_components:
            X.append(ybatch[:, :, i])

        Y = []
        for i in self.output_components:
            Y.append(ybatch[:, :, i])

        if self.use_tensorboard:
            with self.file_writer.as_default():
                tf.summary.image(
                    "Training data", X[0, 0, 0], step=0, max_outputs=16)
        return X, Y

    def __len__(self):
        return int(self.datalen/self.batch_size)

    def __getitem__(self, index):
        return self.__data_generation(self.gdos)

    def __load_img_and_annotation(self, xlist, ylist, i):
        img, annotation = self.Dataset.load_img_and_annotation(i)
        if img.shape != self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]))
        xlist.append(img)
        ylist.append(annotation)

    def __normalize(self, xbatch):
        if isinstance(xbatch, np.ndarray):
            return xbatch/255
        else:
            return np.array(xbatch)/255
