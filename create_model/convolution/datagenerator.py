import numpy as np
import autolib
import keras
import cv2
from glob import glob
from reorder_dataset import get_speed

class image_generator(keras.utils.Sequence):
    def __init__(self, gdos, Dataset, datalen, batch_size, frc, weight_acc=0.5, augm=True, proportion=0.15, flip=True, smoothing=0.1, label_rdm=0, shape=(160,120,3), n_classes=5, memory=49, seq=False, load_speed=(False, False)):
        self.shape = shape
        self.augm = augm
        self.img_cols = shape[0]
        self.img_rows = shape[1]
        self.batch_size = batch_size
        self.gdos = gdos
        self.Dataset = Dataset


        self.frc = frc
        self.weight_acc = weight_acc

        self.n_classes = n_classes
        self.memory_size = memory+1
        self.datalen = datalen
        self.seq = seq

        self.proportion = proportion
        self.smoothing = smoothing
        self.label_rdm = label_rdm
        self.flip = flip
        self.load_speed = load_speed 


    def __data_generation(self, gdos):
        batchfiles = np.random.choice(gdos, size=self.batch_size)
        xbatch = []
        ybatch = []

        for i in batchfiles:
            try:
                img = self.Dataset.load_image(i) 
                img = cv2.resize(img, (self.img_cols, self.img_rows))

                annotations = self.Dataset.load_annotation(i)

                xbatch.append(img)
                ybatch.append(annotations)

            except:
                print(i)

        if self.augm == True:
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

            X_aug, Y_aug = autolib.generate_functions(xbatch, ybatch, proportion=self.proportion)
            
            xbatch = np.concatenate((xbatch, X_aug))
            ybatch = np.concatenate((ybatch, Y_aug))
            
        if self.flip:
            xflip, yflip = autolib.generate_horizontal_flip(xbatch, ybatch, proportion=1)
            xbatch = np.concatenate((xbatch, xflip))
            ybatch = np.concatenate((ybatch, yflip))
        
        weight = autolib.get_weight(ybatch, self.frc, False, acc=self.weight_acc)
        X = [xbatch/255]
        Y = [ybatch[:, 0]]
        weights = [weight]

        if self.load_speed[0]:
            X.append(ybatch[:, 1])

        if self.load_speed[1]:
            Y.append(ybatch[:, 2])
            weights.append(weight)

        return X, Y, weights

    def __len__(self):
        return int(self.datalen/self.batch_size)

    def __getitem__(self, index):
        return self.__data_generation(self.gdos)


