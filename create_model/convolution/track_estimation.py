import math
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

import autolib
import reorder_dataset

class pos_map():
    def __init__(self, size=(512, 512), its=30, steer_coef=30):
        self.pmap = np.zeros((size[0], size[1], 3))
        self.its = its
        self.dt = 1/self.its
        self.steer_coef = steer_coef

        self.angle = 0

    def get_pos(self, paths, Y, speed=1):
        degs = []
        pos_list = []
        pos = [0, 0]
        angle = 0

        for p, ny in zip(paths, Y):
            average = 0
            coef = [1, 0.58, 0, -0.58, -1]

            for it, nyx in enumerate(ny):
                average += nyx*coef[it]

            degree_angle = average*self.steer_coef*self.dt
            rad = math.radians(degree_angle)

            angle += rad
            angle = angle%(2*math.pi)

            x = math.sin(angle)*(speed*self.dt) #speed is here an unknown variable TODO: speed estimation
            y = math.cos(angle)*(speed*self.dt)
            pos[0] += x
            pos[1] += y

            # print(x, y, dt) # debuging

            pos_list.append((pos[0], pos[1]))
            degs.append(degree_angle)

        return pos_list, degs

    def draw(self, paths, pos_list, degs):
        maxp = np.max(pos_list)
        minp = np.min(pos_list)

        h, w, _ = self.pmap.shape
        for it, p in enumerate(pos_list):
            rpx = ((p[0]-minp)/(maxp-minp))*w
            rpy = ((p[1]-minp)/(maxp-minp))*h
            if it==0:
                color = (0, 0, 1)
                th = 3
            elif degs[it]>0:
                color = (1, 0, 0)
                th = 1
            elif degs[it]<0:
                color = (0, 0, 1)
                th = 1
            else:
                color = (0, 1, 0)
                th = 1

            cv2.circle(self.pmap, (int(rpx), int(rpy)), 1, color, thickness=th)
            img = cv2.imread(paths[it])
            
            cv2.imshow('img', img)
            cv2.imshow('pmap', self.pmap)
            cv2.waitKey(1)

        cv2.imshow('pmap', self.pmap)
        cv2.waitKey(0)
    
    def boundaries(self, pos_list):
        iner = []
        outer = []

        offset = 10

        for p in pos_list:
            x, y = p
            iner.append((x, y))


        return
        

if __name__ == "__main__":
    # X = load_dataset()
    # Y = get_preds(X)
    dts, datalen = reorder_dataset.load_dataset('C:\\Users\\maxim\\datasets\\1\\', recursive=False)
    
    ts = date = reorder_dataset.get_date(dts[0])
    te = date = reorder_dataset.get_date(dts[-1])
    dt = te-ts

    datalen = len(dts)
    its = datalen/dt


    Y = []
    for d in dts:
        lab = autolib.get_label(d, flip=False)[0]
        date = reorder_dataset.get_date(d)
        Y.append(lab)

    Y = autolib.label_smoothing(Y, 5, 0) # to categorical

    pmap = pos_map(its=its, steer_coef=35)
    pos_list, degs = pmap.get_pos(dts, Y)
    pmap.draw(dts, pos_list, degs)
    # pmap.boundaries()
