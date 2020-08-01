import numpy as np
import autolib
import keras
import cv2
from glob import glob
from reorder_dataset import get_speed

class image_generator(keras.utils.Sequence):
    def __init__(self, gdos, Dataset, datalen, batch_size, frc, sequence=False, seq_batchsize=64, weight_acc=0.5, augm=True, proportion=0.15, flip=True, smoothing=0.1, label_rdm=0, shape=(160,120,3), n_classes=5, load_speed=(False, False)):
        self.shape = shape
        self.augm = augm
        self.img_cols = shape[0]
        self.img_rows = shape[1]
        self.batch_size = batch_size
        self.gdos = gdos
        self.Dataset = Dataset

        self.frc = frc
        self.weight_acc = weight_acc

        self.sequence = sequence
        self.datalen = datalen

        self.proportion = proportion
        self.smoothing = smoothing
        self.label_rdm = label_rdm
        self.flip = flip
        self.load_speed = load_speed
        self.seq_batchsize = seq_batchsize

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
                    img, annotations = self.Dataset.load_img_and_annotation(path)
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
            # here are the old functions
            # X_bright, Y_bright = autolib.generate_brightness(xbatch, ybatch)
            # X_gamma, Y_gamma = autolib.generate_low_gamma(xbatch, ybatch)
            # X_night, Y_night = autolib.generate_night_effect(xbatch, ybatch)
            # X_shadow, Y_shadow = autolib.generate_random_shadows(xbatch, ybatch)
            # X_chain, Y_chain = autolib.generate_chained_transformations(xbatch, ybatch)
            # X_noise, Y_noise = autolib.generate_random_noise(xbatch, ybatch)
            # X_rev, Y_rev = autolib.generate_inversed_color(xbatch, ybatch)
            # X_glow, Y_glow = autolib.generate_random_glow(xbatch, ybatch)
            # X_cut, Y_cut = autolib.generate_random_cut(xbatch, ybatch)

            """ # this is the old "bourrin" way where we add transformed image to the clean one
            X_aug, Y_aug = autolib.generate_functions(xbatch, ybatch, proportion=self.proportion)

            xbatch = np.concatenate((xbatch, X_aug))
            ybatch = np.concatenate((ybatch, Y_aug))
            """

            # this is much nicer as we modify into the batch of clean image
            if self.sequence:
                for i in range(len(xbatch)):
                    xbatch[i], ybatch[i] = autolib.generate_functions_replace(xbatch[i], ybatch[i],
                                                                              proportion=self.proportion)
            else:
                xbatch, ybatch = autolib.generate_functions_replace(xbatch, ybatch,
                                                                    proportion=self.proportion)

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

            if self.load_speed[0]:
                X = [xbatch, ybatch[:, :, 1]]
            else:
                X = xbatch

            if self.load_speed[1]:
                Y = [ybatch[:, :, 0], ybatch[:, :, 2]]
            else:
                Y = ybatch[:, :, 0]

            Y = np.expand_dims(Y, axis=-1)

        else:

            if self.load_speed[0]:
                X = [xbatch, ybatch[:, 1]]
            else:
                X = xbatch

            if self.load_speed[1]:
                Y = [ybatch[:, 0], ybatch[:, 2]]
            else:
                Y = ybatch[:, 0]

        return X, Y

    def __len__(self):
        return int(self.datalen/self.batch_size)

    def __getitem__(self, index):
        return self.__data_generation(self.gdos)
