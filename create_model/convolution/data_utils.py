import os
import time
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
import shutil

import reorder_dataset
import autolib


class Data(): # TODO: clean data class (could be used elsewhere)
    def __init__(self, dos, is_float=True, recursive=False):
        self.dos = dos
        self.is_float = is_float
        self.recursive = recursive
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


    def img_name_format(self, dos, lab, path):
        str_name = dos+str(lab)
        img_name_component = path.split('\\')[-1].split('_')[1:]
        for info in img_name_component:
            str_name += '_'+str(info)
        return str_name

    def save(self, dts, Y=[], name="saved", mode=0):
        new_dos = self.dos+"..\\"+name+"\\"
        if os.path.isdir(new_dos) == False:
            os.mkdir(new_dos)

        if mode == 0: # save image with a new label
            for path, y in tqdm(zip(dts, Y)):
                time.sleep(0.0001)
                name = self.img_name_format(new_dos, y, path)
                shutil.copy(path, name)

        else: # copy image by loading it and resaving it 
            for path in tqdm(dts):
                time.sleep(0.0001)
                new_path = new_dos+path.split('\\')[-1]
                shutil.copy(path, new_path)

def catlab2linear(lab, dico=[3, 5, 7, 9, 11]):
    return (dico.index(lab)-2)/2

def lab2linear_smooth(lab, cat2linear=False, window_size=(0, 5), sq_factor=1, prev_factor=1, after_factor=1, offset=0):
    if cat2linear:
        linear = [catlab2linear(i) for i in lab]
    else:
        linear = lab
    smooth = average_data(linear, window_size=window_size, sq_factor=sq_factor, prev_factor=prev_factor, after_factor=after_factor, offset=offset)
    return smooth

def offset_data(data, offset):
    data[0:-offset] = data[offset:-1]
    return data

def average_data(data, window_size=(5, 5), sq_factor=1, prev_factor=1, after_factor=1, offset=0):
    averaged = []
    weights = ([prev_factor]*window_size[0])+([after_factor]*window_size[1])

    for i in range(window_size[0], len(data)-window_size[1]-offset):
        av = np.average(data[(i+offset)-window_size[0]: (i+offset)+window_size[1]], axis=-1)
        if av>=0:
            av = av**sq_factor
        else:
            av = -np.absolute(av)**sq_factor

        averaged.append(av)

    data[offset+window_size[0]:-window_size[1]] = averaged
    return data

def detect_spike(labs, th=0.5, window_size=(5, 5), offset=5):
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

def detect_straight(labs, th=0.1, window_size=10):
    straights = []
    for lab in labs:
        if np.absolute(lab)<th:
            straights.append(1)
        else:
            straights.append(0)

    return straights

def get_timetoclosestturn(X, spikes):
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

def get_accbrake_periods(straights, times, time_threshold=3):
    variations = []
    for s, t in zip(straights, times):
        if t<time_threshold:
            variations.append(1)
        # elif s==1:
        #     variations.append(1)
        else:
            variations.append(0)
    return variations

def load_lab(dts, is_float=True):
    X = []
    labs = []
    for path in dts:
        lab = path.split('\\')[-1].split('_')[0]
        if is_float:
            lab = float(lab)
        else:
            lab = int(lab)
        X.append(path)
        labs.append(lab)
    return np.array(X), np.array(labs)

if __name__ == "__main__":
    # quick code to transform images from categorical to linear mode
    root_dos = "C:\\Users\\maxim\\random_data\\"
    save_dos = "linear"

    doss = ["C:\\Users\\maxim\\random_data\\17 custom maps"]
    # doss = "C:\\Users\\maxim\\random_data\\"
    reccurent = False

    if reccurent:
        doss = glob(doss+"*")
    
    for dos in doss:
        if dos.split('\\')[-1] != save_dos:
            is_float = False
            try: 
                p = glob(dos+"\\*")[0]
                d = autolib.get_label(p)
            except:
                is_float=True
            cat2linear = not is_float

            data = Data(dos+"\\", is_float=is_float, recursive=False)
            dts, Y = data.load_lab()
            Y = lab2linear_smooth(Y, cat2linear=cat2linear, window_size=(0,5), sq_factor=1, prev_factor=1, after_factor=1, offset=1)
            data.save(dts, Y, name=save_dos+"\\"+dos.split('\\')[-1])