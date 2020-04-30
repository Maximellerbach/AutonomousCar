import math
import time
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

import architectures
import autolib
import reorder_dataset


def average_data(data, window_size=10, sq_factor=1):
    averaged = []
    for i in range(window_size//2, len(data)-window_size//2):
        averaged.append(np.average(data[i-window_size//2: i+window_size//2], axis=-1)**sq_factor)

    data[window_size//2:-window_size//2] = averaged

    return data

def compare_pred(self, dos='C:\\Users\\maxim\\datasets\\1 ironcar driving\\', dt_range=(0, -1)):
    paths, dts_len = reorder_dataset.load_dataset(dos, recursive=False)
    paths = paths[dt_range[0]:dt_range[1]]
    dts_len = len(paths)

    X = []
    Y = []
    for path in tqdm(paths):
        lab = autolib.get_label(path, flip=False)[0]

        Y.append(lab)
        X.append(cv2.imread(path)/255)

    X = np.array(X)
    Y = to_categorical(Y)
    Y = architectures.cat2linear(Y)

    # Y = average_data(Y, window_size=10)

    pred_Y = self.model.predict(X)
    pred_Y = architectures.cat2linear(pred_Y)

    plt.plot([i for i in range(dts_len)], Y, pred_Y)
    plt.show()


def evaluate_speed(self, data_path='C:\\Users\\maxim\\datasets\\1 ironcar driving\\'):
    paths = glob(data_path+"*")
    X = np.array([cv2.resize(cv2.imread(i), (160,120)) for i in tqdm(paths[:5000])])

    st = time.time()
    preds = self.model.predict(X/255)
    et = time.time()
    dt = et-st

    pred_dt = dt/len(X)
    frc = 1/pred_dt

    return (dt, pred_dt, frc)


def pred_img(self, img, size, sleeptime, nimg_size=(5, 5)):
    """
    predict an image and visualize the prediction
    """
    img = cv2.resize(img, size)
    pred = np.expand_dims(img/255, axis=0)

    nimg = self.fe.predict(pred)[0]
    nimg = np.expand_dims(cv2.resize(nimg, nimg_size), axis=0)
    n = nimg.shape[-1]

    if self.recurrence == True:
        filled = [[0, 0.125, 0.75, 0.125, 0]]*(self.memory_size-len(self.av))+self.av
        rec = np.expand_dims(filled, axis=0)
        # print(pred.shape, rec.shape)
        ny = self.model.predict([pred, rec])[0]
    else:
        ny = self.model.predict(pred)[0]

    lab = np.argmax(ny)
    
    # average softmax direction
    average = architectures.cat2linear([ny])[0] # here you convert a list of cat to a list of linear
    ny = [round(i, 3) for i in ny]
    # print(ny, average)

    
    if len(self.av)<self.memory_size:
        self.av.append(ny)
    else:
        self.av.append(ny)
        del self.av[0]

    square_root = int(math.sqrt(n))+1
    tot_img = np.zeros((nimg.shape[1]*square_root, nimg.shape[2]*square_root))

    try:
        for x in range(square_root):
            for y in range(square_root):
                tot_img[nimg.shape[1]*x:nimg.shape[1]*(x+1), nimg.shape[2]*y:nimg.shape[2]*(y+1)] = (nimg[0, :, :, x*square_root+y])
    except:
        pass

    c = np.copy(img)
    cv2.line(c, (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+average*30), img.shape[0]-50), color=[255, 0, 0], thickness=4)
    c = c/255

    if n==1:
        av = nimg[0]
        av = cv2.resize(av, size)
        cv2.imshow('im', nimg[0, :, :])
    else:
        av = np.sum(nimg[0], axis=-1)
        av = cv2.resize(av/(nimg.shape[-1]/2), size)
        cv2.imshow('tot', tot_img)
    cv2.imshow('img', c)
    cv2.waitKey(sleeptime)

def load_frames(self, path, size=(160, 120), batch_len=32):
    """
    load a batch of frame from video
    """
    batch = []
    
    for _ in range(batch_len):
        _, frame = self.cap.read()
        frame = cv2.resize(frame, size)
        batch.append(frame)

    return batch

def after_training_test_pred(self, path='C:\\Users\\maxim\\random_data\\4 trackmania A04\\*', size=(160,120), nimg_size=(5,5), from_path=True, batch_vid=32, sleeptime=1):
        """
        either predict images in a folder
        or from a video
        """

        if from_path==True:
            for i in glob(path):
                img = cv2.imread(i)
                pred_img(self, img, size, sleeptime, nimg_size=nimg_size)
                
        else:
            self.cap = cv2.VideoCapture(path)
            print("created cap")

            while(True):
                im_batch = self.load_frames(path, batch_len=batch_vid)
                for img in im_batch:
                    pred_img(self, img, size, sleeptime, nimg_size=nimg_size)



def process_trajectory_error(): # TODO: evaluate long term precision of the model 
    return
