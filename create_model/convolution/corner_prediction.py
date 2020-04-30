import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten)
from keras.models import Input, Model, load_model
from keras.optimizers import Adam
from glob import glob

import reorder_dataset
from data_class import data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default

def create_brake_model(model_name="test_model\\convolution\\fe.h5"):
    model = load_model(model_name)
    model.trainable = True

    inp = Input((120, 160, 3))
    y = model(inp)
    y = Flatten()(y)

    y = Dense(50, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("linear")(y)
    y = Dropout(0.2)(y)

    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("linear")(y)
    y = Dropout(0.1)(y)

    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("linear")(y)
    y = Dropout(0.1)(y)

    y = Dense(1, activation="tanh", use_bias=False)(y)

    new_model = Model(inp, y)
    new_model.compile(Adam(0.001), loss="mse")
    new_model.summary()

    return new_model

def train_model(X, Y, batch_size=32, epochs=2):
    new_model = create_brake_model()

    new_model.fit(X, Y, batch_size=batch_size, epochs=epochs)
    new_model.save("test_model\\convolution\\acc_brake.h5")
    return new_model

def test_pred(model, X, dos = "C:\\Users\\maxim\\recorded_imgs\\0_0_1588152155.6270337\\"):
    for path in glob(dos+"*"):
        img = cv2.imread(path)/255
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        cv2.imshow('img', img)
        print(pred)
        cv2.waitKey(0)

class make_labels(data): # TODO: look further in the data to make labels
    def __init__(self, dos="C:\\Users\\maxim\\recorded_imgs\\0\\", is_float=True, show=False):
        super().__init__(dos, is_float=is_float)
        window_size = 10

        self.X, self.labs = self.load_lab()
        original_labs = self.labs

        self.labs = self.average_data(self.labs, window_size=window_size)
        self.spikes = self.detect_spike(self.labs, th=0.4, window_size=window_size, offset=0)

        self.times, end = self.get_timetoclosestturn(self.X, self.spikes)
        self.straigts = self.detect_straight(self.labs, th=0.02)
        self.variations = self.get_accbrake_periods(self.straigts, self.times, time_threshold=2)
        self.variations = self.average_data(self.variations, window_size=window_size, sq_factor=1)

        self.labs = self.labs[:end]
        self.X = self.X[:end]
        
        if show:
            plt.plot([i for i in range(len(self.labs))], self.labs, self.variations)
            plt.show()

    def train(self):
        imgs = self.load_img(self.X)
        model = train_model(imgs, self.variations, batch_size=16, epochs=30)
        print("trained model")
    
    def test(self):
        model = load_model("test_model\\convolution\\acc_brake.h5")
        imgs = self.load_img(self.X)
        test_pred(model, imgs)

if __name__ == "__main__":
    brake_data = make_labels(dos="C:\\Users\\maxim\\recorded_imgs\\0_0_1588152155.6270337\\", is_float=True, show=True)
    brake_data.train()
    brake_data.test()
    
