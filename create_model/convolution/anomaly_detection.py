import numpy as np
import keras.backend as K
from keras.models import load_model, Model
from keras.losses import mse, mae
from keras.layers import *

import tensorflow as tf

import cv2
from glob import glob

import architectures
from datagenerator import image_generator
import reorder_dataset


class anomaly_AE():
    def __init__(self, name, dospath='', load=True):
        self.name = name
        self.dospath = dospath

        if load == True:
            try:
                self.decoder = load_model(self.name)
            except:
                print('cannot load model, creating it.')
                self.encoder, self.decoder = self.create_model()
        else:
            self.encoder, self.decoder = self.create_model()

    def create_model(self):
        input_shape = (120,160,3)
        _, encoder = architectures.create_light_CNN(input_shape, 5)
        inp = Input(input_shape)
        y = encoder(inp)

        y = Conv2DTranspose(48, kernel_size=(3,3), strides=(8,2), use_bias=False, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = Conv2DTranspose(48, kernel_size=(3,3), strides=(2,2), use_bias=False, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        
        y = Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), use_bias=False, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = Conv2DTranspose(16, kernel_size=(5,5), strides=(2,2), use_bias=False, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = Conv2DTranspose(8, kernel_size=(5,5), strides=(2,2), use_bias=False, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Cropping2D((4, 0))(y)
        
        y = Conv2D(3, kernel_size=1, strides=1, use_bias=False, padding='same', activation='sigmoid')(y)

        decoder = Model(inp, y)
        decoder.compile('adam', mse+mae, metrics=['mae', 'binary_crossentropy'])
        decoder.summary()

        return encoder, decoder

    def train_model(self, batch_size=32, epochs=10):
        self.epochs = epochs

        self.gdos, self.datalen = reorder_dataset.load_dataset(self.dospath)
        # self.gdos = np.concatenate([i for i in self.gdos])
        self.gdos = self.gdos[1]
        self.datalen = len(self.gdos)
        np.random.shuffle(self.gdos)
        self.gdos, self.valdos = np.split(self.gdos, [self.datalen-self.datalen//20])

        self.decoder.fit_generator(image_generator(self.gdos, self.datalen, batch_size, augm=True, reconstruction=True), steps_per_epoch=(self.datalen//batch_size), epochs=self.epochs, 
                                                    validation_data=image_generator(self.valdos, self.datalen, batch_size, augm=True, reconstruction=True), validation_steps=(self.datalen//20)//batch_size, max_queue_size=5, workers=8)

        self.decoder.save(self.name)

    def detect_anomaly(self, dos):
        anomalies = []
        paths = glob(dos)
        for it, path in enumerate(paths):
            img = cv2.imread(path)
            img = cv2.resize(img, (160, 120))/255
            to_pred = np.expand_dims(img, axis=0)

            pred = self.decoder.predict(to_pred)

            # mse_loss = K.mean(mse(to_pred, pred))
            # mae_loss = K.mean(mae(to_pred, pred))

            # av_mse = K.eval(mse_loss)
            # av_mae = K.eval(mae_loss)

            av_mse = np.sqrt(np.mean((pred - to_pred) ** 2))

            if av_mse>0.15:
                print(it, av_mse)
                anomalies.append(path)

            cv2.imshow('img', img)
            cv2.imshow('pred', pred[0])
            cv2.waitKey(1)

        print(len(paths), len(anomalies))

if __name__ == "__main__":
    AE = anomaly_AE("anomaly.h5", dospath='C:\\Users\\maxim\\datasets\\*', load=True)
    # AE.train_model(batch_size=8, epochs=2)
    AE.detect_anomaly('C:\\Users\\maxim\\datasets\\4\\*')
    AE.detect_anomaly('C:\\Users\\maxim\\datasets\\2\\*')zdq
