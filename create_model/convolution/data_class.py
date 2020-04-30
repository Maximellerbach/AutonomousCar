import cv2
import numpy as np

import reorder_dataset
import matplotlib.pyplot as plt


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
                lab = self.catlab2linear(lab)

            X.append(path)
            labs.append(lab)
        return np.array(X), np.array(labs)

    def catlab2linear(self, lab, dico=[3, 5, 7, 9, 11]):
        return (dico.index(lab)-2)/2

    def catlab2linear_smooth(self, lab, window_size=10, sq_factor=1):
        linear = self.catlab2linear(lab)
        return self.average_data(linear, window_size=window_size, sq_factor=sq_factor)

    def average_data(self, data, window_size=10, sq_factor=1):
        averaged = []
        for i in range(window_size//2, len(data)-window_size//2):
            averaged.append(np.average(data[i-window_size//2: i+window_size//2], axis=-1)**sq_factor)

        index_modifier = 0
        data[window_size//2:-window_size//2] = averaged

        return data

    def detect_spike(self, labs, th=0.5, window_size=10, offset=5):
        spikes = []
        spike = []
        is_spike = False
        for it, lab in enumerate(labs):
            if lab>=th and is_spike == False:
                spike.append(it-window_size//2-offset)
                is_spike = True

            elif lab<th and is_spike == True:
                spike.append(it+window_size//2-offset)
                is_spike = False
                spikes.append(spike)
                spike = []

        return spikes

    def detect_straight(self, labs, th=0.1, window_size=10):
        straights = []
        for lab in labs:
            if lab<th:
                straights.append(1)
            else:
                straights.append(0)

        return straights

    def get_timetoclosestturn(self, X, spikes):
        def get_next_spike(spikes, index):
            it = 0
            if len(spikes)<0:
                print("select lower threshold value")
                return None
            if index>spikes[-1][0]:
                return None

            while(index>spikes[it][0]):
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


    def get_accbrake_periods(self, straights, times, time_threshold=3):
        variations = []
        for s, t in zip(straights, times):
            if t<time_threshold:
                variations.append(-1)
            elif s==1:
                variations.append(1)
            else:
                variations.append(0)
        return variations

