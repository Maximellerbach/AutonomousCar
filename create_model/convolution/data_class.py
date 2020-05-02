import os
import time
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

import reorder_dataset


class Data(): # TODO: clean data class (could be used elsewhere)
    def __init__(self, dos, is_float=True, recursive=False):
        self.dos = dos
        self.is_float = is_float
        self.dts, self.datalen = reorder_dataset.load_dataset(self.dos, recursive=recursive)

    def load_img(self, path):
        return cv2.imread(path)

    def load_imgs(self, paths):
        imgs = []
        for path in tqdm(paths):
            img = self.load_img(path) 
            imgs.append(img)

        return np.array(imgs)

    def load_lab(self):
        X = []
        labs = []
        for path in self.dts:
            lab = path.split('\\')[-1].split('_')[0]
            if self.is_float:
                lab = float(lab)
            else:
                lab = int(lab)

            X.append(path)
            labs.append(lab)
        return np.array(X), np.array(labs)

    def catlab2linear(self, lab, dico=[3, 5, 7, 9, 11]):
        return (dico.index(lab)-2)/2

    def catlab2linear_smooth(self, lab, window_size=(0, 5), sq_factor=1):
        linear = [self.catlab2linear(i) for i in lab]
        smooth = self.average_data(linear, window_size=window_size, sq_factor=sq_factor)
        return smooth

    def average_data(self, data, window_size=(5, 5), sq_factor=1):
        averaged = []
        for i in range(window_size[0], len(data)-window_size[1]):
            averaged.append(np.average(data[i-window_size[0]: i+window_size[1]], axis=-1)**sq_factor)

        data[window_size[0]:-window_size[1]] = averaged

        return data

    def detect_spike(self, labs, th=0.5, window_size=(5, 5), offset=5):
        spikes = []
        spike = []
        is_spike = False
        for it, lab in enumerate(labs):
            if np.absolute(lab)>=th and is_spike == False:
                spike.append(it-window_size[0]-offset)
                is_spike = True

            elif np.absolute(lab)<th and is_spike == True:
                spike.append(it+window_size[1]-offset)
                is_spike = False
                spikes.append(spike)
                spike = []

        return spikes

    def detect_straight(self, labs, th=0.1, window_size=10):
        straights = []
        for lab in labs:
            if np.absolute(lab)<th:
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
                variations.append(1)
            # elif s==1:
            #     variations.append(1)
            else:
                variations.append(0)
        return variations

    def img_name_format(self, dos, lab, format=".png"):
        return dos+str(lab)+"_"+str(time.time())+".png"

    def save(self, dts, Y, name="saved"):
        new_dos = self.dos+"..\\"+name+"\\"
        try:
            os.mkdir(new_dos)
        except:
            pass

        for path, y in tqdm(zip(dts, Y)):
            x = self.load_img(path)
            name = self.img_name_format(new_dos, y)
            cv2.imwrite(name, x)

if __name__ == "__main__":

    root_dos = "C:\\Users\\maxim\\random_data\\"
    doss = glob(root_dos+'*')
    save_dos = "linear"

    for i, dos in enumerate(doss):
        if dos.split('\\')[-1] != save_dos:
            data = Data(dos+"\\", is_float=False, recursive=False)
            dts, Y = data.load_lab()
            Y = data.catlab2linear_smooth(Y, window_size=(5,5))
            data.save(dts, Y, name=save_dos+"\\"+dos.split('\\')[-1])
