import numpy as np
import autolib
import keras
import cv2

class image_generator(keras.utils.Sequence):
    def __init__(self, img_path, batch_size=32, augm=True, shape=(160,120,3), n_classes=5):
        self.shape = shape
        self.augm = augm
        self.img_cols = shape[0]
        self.img_rows = shape[1]
        self.batch_size = batch_size
        self.img_path = img_path
        self.n_classes = n_classes

    def __data_generation(self, img_path):
        batchfiles = np.random.choice(img_path, size=self.batch_size)
        xbatch = []
        ybatch = []

        for i in batchfiles:
            img = cv2.imread(i)
            img = cv2.resize(img, (self.img_cols, self.img_rows))
            # img = autolib.cut_img(img, 20)

            label = autolib.get_label(i, flip=False, before=True, reg=False) #reg=True for [-1;1] directions

            xbatch.append(img)
            ybatch.append(label[0])

        xflip, yflip = autolib.generate_horizontal_flip(xbatch, ybatch, proportion=1)
        xbatch = np.concatenate((xbatch, xflip))
        ybatch = np.concatenate((ybatch, yflip))

        if self.augm == True:
            X_bright, Y_bright = autolib.generate_brightness(xbatch, ybatch, proportion=0.25)
            X_gamma, Y_gamma = autolib.generate_low_gamma(xbatch, ybatch, proportion=0.25)
            X_night, Y_night = autolib.generate_night_effect(xbatch, ybatch, proportion=0.25)
            X_shadow, Y_shadow = autolib.generate_random_shadows(xbatch, ybatch, proportion=0.25)
            X_chain, Y_chain = autolib.generate_chained_transformations(xbatch, ybatch, proportion=0.25)
            X_noise, Y_noise = autolib.generate_random_noise(xbatch, ybatch, proportion=0.25)
            X_rev, Y_rev = autolib.generate_inversed_color(xbatch, ybatch, proportion=0.25)
            X_glow, Y_glow = autolib.generate_random_glow(xbatch, ybatch, proportion=0.25)

            xbatch = np.concatenate((xbatch, X_gamma, X_bright, X_night, X_shadow, X_chain, X_noise, X_rev, X_glow))/255
            ybatch = np.concatenate((ybatch, Y_gamma, Y_bright, Y_night, Y_shadow, Y_chain, Y_noise, Y_rev, Y_glow))
        else:
            xbatch = xbatch/255

        # ybatch = to_categorical(ybatch, self.number_class)
        ybatch = autolib.label_smoothing(ybatch, 5, 0.25)

        return xbatch, ybatch

    def __len__(self):
        return int(len(self.img_path)/self.batch_size)

    def __getitem__(self, index):
        X, Y = self.__data_generation(self.img_path)
        return X, Y