import time
from glob import glob

import cv2
import keras
import numpy as np

import autolib


class image_generator(keras.utils.Sequence):
    def __init__(self, gdos, Dataset, input_components, output_components, datalen, batch_size, frc,
                 sequence=False, seq_batchsize=64, augm=True, proportion=0.15,
                 flip=True, smoothing=0.1, label_rdm=0, n_classes=5, lab_scale=1, lab_bias=0,
                 shape=(160, 120, 3)):
        self.shape = shape
        self.augm = augm
        self.img_cols = shape[0]
        self.img_rows = shape[1]
        self.batch_size = batch_size
        self.gdos = gdos
        self.Dataset = Dataset

        self.frc = frc
        self.lab_scale = lab_scale
        self.lab_bias = lab_bias

        self.input_components = input_components
        self.output_components = output_components
        self.sequence = sequence
        self.datalen = datalen

        self.proportion = proportion
        self.smoothing = smoothing
        self.label_rdm = label_rdm
        self.flip = flip
        self.seq_batchsize = seq_batchsize
        self.creation_time = time.time()

    def __data_generation(self, gdos):
        batchfiles = np.random.choice(gdos, size=self.batch_size)
        xbatch = []
        ybatch = []

        for i in batchfiles:
            if self.sequence:
                xseq = []
                yseq = []

                seq_len = len(i)
                if seq_len > self.seq_batchsize:
                    rdm_seq = np.random.randint(0, seq_len-self.seq_batchsize)
                    i = i[rdm_seq:rdm_seq+self.seq_batchsize]

                elif seq_len < self.seq_batchsize:  # ugly way to make sure every sequence has the same length
                    i = [i[0]]*(self.seq_batchsize-seq_len)+i

                for path in i:
                    img, annotations = self.Dataset.load_img_and_annotation(
                        path)
                    img = cv2.resize(img, (self.img_cols, self.img_rows))

                    xseq.append(img)
                    yseq.append(annotations)

                xbatch.append(xseq)
                ybatch.append(yseq)

            else:
                img, annotations = self.Dataset.load_img_and_annotation(i)
                img = cv2.resize(img, (self.img_cols, self.img_rows))
                xbatch.append(img)
                ybatch.append(annotations)

        if self.augm:
            """ # this is the old "bourrin" way where we add transformed image to the clean one
            X_aug, Y_aug = autolib.generate_functions(
                xbatch, ybatch, proportion=self.proportion)

            xbatch = np.concatenate((xbatch, X_aug))
            ybatch = np.concatenate((ybatch, Y_aug))
            """

            # this is much nicer as we modify into the batch of clean image
            if self.sequence:
                for i in range(len(xbatch)):
                    xbatch[i], ybatch[i] = autolib.generate_functions_replace(xbatch[i], ybatch[i],
                                                                              proportion=self.proportion,
                                                                              functions=(autolib.change_brightness,
                                                                                         autolib.rescut,
                                                                                         autolib.inverse_color,
                                                                                         autolib.add_random_shadow,
                                                                                         autolib.add_random_glow,
                                                                                         autolib.rdm_noise))
            else:
                xbatch, ybatch = autolib.generate_functions_replace(xbatch, ybatch,
                                                                    proportion=self.proportion,
                                                                    functions=(autolib.change_brightness,
                                                                               autolib.rescut,
                                                                               autolib.inverse_color,
                                                                               autolib.add_random_shadow,
                                                                               autolib.add_random_glow,
                                                                               autolib.rdm_noise))

        if self.flip:
            if self.sequence:
                for i in range(len(xbatch)):
                    xflip, yflip = autolib.generate_horizontal_flip(xbatch[i], ybatch[i],
                                                                    proportion=1)
                    xbatch.append(xflip)
                    ybatch.append(yflip)

                xbatch = np.array(xbatch)/255
                ybatch = np.array(ybatch)

            else:
                xflip, yflip = autolib.generate_horizontal_flip(xbatch, ybatch,
                                                                proportion=1)
                xbatch = np.concatenate((xbatch, xflip))/255
                ybatch = np.concatenate((ybatch, yflip))

        # removed the weight, useless ; weight = autolib.get_weight(ybatch, self.frc, False, acc=self.weight_acc)

        if self.sequence:
            X = [xbatch]
            for i in self.input_components:
                X.append(ybatch[:, :, i])

            Y = []
            for i in self.output_components:
                Y.append(ybatch[:, :, i])

        else:
            X = [xbatch]
            for i in self.input_components:
                X.append(ybatch[:, i])

            Y = []
            for i in self.output_components:
                Y.append(ybatch[:, i])

        return X, Y

    def __len__(self):
        return int(self.datalen/self.batch_size)

    def __getitem__(self, index):
        return self.__data_generation(self.gdos)
