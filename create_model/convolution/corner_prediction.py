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
    y = Activation("relu")(y)
    y = Dropout(0.2)(y)

    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(0.1)(y)

    y = Dense(25, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(0.1)(y)

    y = Dense(1, activation="sigmoid", use_bias=False)(y)

    new_model = Model(inp, y)
    new_model.compile(Adam(0.001), loss="mse")
    new_model.summary()

    return new_model

def train_model(X, Y, batch_size=32, epochs=2):
    new_model = create_brake_model()

    new_model.fit(X, Y, batch_size=batch_size, epochs=epochs)
    new_model.save("test_model\\convolution\\brakev6_mix.h5")
    return new_model

def test_pred(model, X, dos = 'C:\\Users\\maxim\datasets\\8 sim correction\\'):
    for path in glob(dos+"*"):
        img = cv2.imread(path)/255
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        cv2.imshow('img', img)
        print(pred)
        cv2.waitKey(0)

class data(): # TODO: clean data class (could be used elsewhere)
    def __init__(self, dos, is_float=True):
        self.dos = dos
        self.is_float = is_float

    def load_img(self, dts):
        imgs = []
        for path in dts:
            img = cv2.imread(path)/255
            imgs.append(img)

        return np.array(imgs)

    def load_lab(self):
        X = []
        labs = []
        dts, datalen = reorder_dataset.load_dataset(self.dos, recursive=False)
        for path in dts:
            lab = path.split('\\')[-1].split('_')[0]
            if self.is_float:
                lab = float(lab)
            else:
                lab = int(lab)
                lab = self.transform_lab(lab)

            X.append(path)
            labs.append(lab)
        return np.array(X), np.array(labs)

    def transform_lab(self, lab, dico=[3, 5, 7, 9, 11]):
        return (dico.index(lab)-2)/2

    def average_data(self, data, window_size=10, sq_factor=1):
        averaged = []
        for i in range(window_size//2, len(data)-window_size//2):
            averaged.append(np.average(data[i-window_size//2: i+window_size//2], axis=-1)**sq_factor)

        index_modifier = 0
        data[window_size//2:-window_size//2] = averaged

        return data

    def detect_spike(self, labs, th=0.5, window_size=10):
        spikes = []
        spike = []
        is_spike = False
        for it, lab in enumerate(labs):
            if lab>=th and is_spike == False:
                spike.append(it-window_size//2)
                is_spike = True

            elif lab<th and is_spike == True:
                spike.append(it+window_size//2)
                is_spike = False
                spikes.append(spike)
                spike = []

        return spikes

    def get_timetoclosestturn(self, X, spikes):
        def get_next_spike(spike, index):
            it = 0
            if index>spikes[-1][0]:
                return None

            while(index>spike[it][0]):
                it += 1

            return it

        times = []
        for i in range(len(X)):
            n_spike = get_next_spike(spikes, i)
            if n_spike != None:

                actual_time = reorder_dataset.get_date(X[i])
                next_spike_time = reorder_dataset.get_date(X[spikes[n_spike][0]-10])

                dt = next_spike_time-actual_time
                times.append(dt)

            else:
                return times, i

        return times, i

    def get_brake_periods(self, times, time_threshold=3):
        brakes = []
        for t in times:
            if t<time_threshold:
                brakes.append(1)
            else:
                brakes.append(0)
        return brakes


class make_labels(data): # TODO: look further in the data to make labels
    def __init__(self, dos="C:\\Users\\maxim\\recorded_imgs\\0\\", is_float=True, show=False):
        super().__init__(dos, is_float=is_float)
        window_size = 10

        self.X, self.labs = self.load_lab()
        original_labs = self.labs

        self.labs = self.average_data(self.labs, window_size=window_size)
        spikes = self.detect_spike(self.labs, th=0.7, window_size=window_size)

        self.times, end = self.get_timetoclosestturn(self.X, spikes)
        self.brakes = self.get_brake_periods(self.times, time_threshold=2)
        self.brakes = self.average_data(self.brakes, window_size=6, sq_factor=1)

        self.labs = self.labs[:end]
        self.X = self.X[:end]
        
        if show:
            plt.plot([i for i in range(len(self.labs))], self.labs, self.brakes)
            plt.show()

    def train(self):
        imgs = self.load_img(self.X)
        model = train_model(imgs, self.brakes, batch_size=32, epochs=20)
        print("trained model")
        
        model = load_model("test_model\\convolution\\brakev6_mix.h5")
        test_pred(model, imgs)

if __name__ == "__main__":
    brake_data = make_labels(dos="C:\\Users\\maxim\\recorded_imgs\\0_0_1587729884.301688\\", is_float=True, show=True)
    brake_data.train()
    
