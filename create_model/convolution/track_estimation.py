import autolib
import math
from glob import glob

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import matplotlib.pyplot as plt

import reorder_dataset


class pos_map():
    def __init__(self, size=(512, 512), its=30, steer_coef=30):
        self.pmap = np.zeros((size[0], size[1], 3))
        self.its = its
        self.dt = 1/self.its
        self.steer_coef = steer_coef

        self.segment_map = []
        self.angle = 0

    def clear_pmap(self):
        self.pmap = np.zeros_like(self.pmap)

    def normalize_pt(self, p, maxp, minp, entier=True, shape=(512, 512)):
        h, w = shape

        x = ((p[0]-minp)/(maxp-minp))*w
        y = ((p[1]-minp)/(maxp-minp))*h
        if entier == True:
            x = int(x)
            y = int(y)

        return (x, y)

    def get_pos(self, paths, Y, speed=1):
        vect_list = []
        pos_list = []
        minpos_list = []
        degs = []
        pos = [0, 0]
        angle = 0

        for it, ny in enumerate(Y):
            average = 0
            coef = [1, 0.58, 0, -0.58, -1]

            for ait, nyx in enumerate(ny):
                average += nyx*coef[ait]

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
            vect_list.append((x, y))
            degs.append(rad)

            if degree_angle!=0. or it==0:
                minpos_list.append((pos[0], pos[1], it))
            

        return pos_list, minpos_list, vect_list, degs


    def boundaries(self, pos_list, r=1):
        iner = []
        outer = []

        offset = 1
        for p1, p2 in zip(pos_list[:-1], pos_list[1:]):
            x1, y1 = p1[:2]
            x2, y2 = p2[:2]

            way = r if x2-x1>0 else -r

            if (x2-x1) != 0:
                a = (y2-y1)/(x2-x1)

                angle = math.pi/2+math.atan(a)
                wi = math.cos(angle)*offset*way
                hi = math.sin(angle)*offset*way

                iner.append((x1+wi,y1+hi))
                outer.append((x1-wi,y1-hi))

            else:
                iner.append((x1+offset*way, y1))
                outer.append((x1-offset*way, y1))

        return iner, outer


    def segment_track(self, pos_list, deg_list, th=0.005, look_back=30):
        turning = False
        way = 0
        average = [0]*len(deg_list)
        thresholded = []
        turn = []

        for it in range(len(deg_list)):
            if it>=look_back:
                average[it] = np.average(deg_list[it-look_back:it])

                if average[it]>=th:
                    if turning == False:
                        way = 1
                        x, y = pos_list[it-look_back//2]
                        turn.append((x,y,it,way))
                        turning = True

                elif average[it]<=-th:
                    if turning == False:
                        way = -1
                        x, y = pos_list[it-look_back//2]
                        turn.append((x,y,it,way))
                        turning = True

                elif turning == True:
                    x, y = pos_list[it-look_back//2]
                    turn.append((x,y,it,way))
                    thresholded.append(turn)
                    turning = False
                    turn = []
                    way = 0

        return thresholded, average

    def match_segments(self, pos_list):
        return

    def draw_points(self, pos_list, degs=[]):
        maxp = np.max(pos_list)
        minp = np.min(pos_list)

        h, w, _ = self.pmap.shape
        for it, p in enumerate(pos_list):
            rpx = ((p[0]-minp)/(maxp-minp))*w
            rpy = ((p[1]-minp)/(maxp-minp))*h
            try:
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
            except:
                color = (0, 1, 0)
                th = 1


            cv2.circle(self.pmap, (int(rpx), int(rpy)), 1, color, thickness=th)
    

    def draw_segments(self, segments, min_max=True):
        segments = np.array(segments)
        segments_points = segments[:, :, :2]
        if min_max==True:
            maxp = np.max(segments_points)
            minp = np.min(segments_points)
        else:
            maxp = np.max(min_max)
            minp = np.min(min_max)

        for segm in segments:
            p1 = self.normalize_pt(segm[0], maxp, minp)
            p2 = self.normalize_pt(segm[1], maxp, minp)
            way = segm[0][-1]

            if way>0:
                color = (1, 0, 0)
            elif way<0:
                color = (0, 0, 1)

            cv2.line(self.pmap, p1, p2, color, thickness=3)

if __name__ == "__main__":
    dts, datalen = reorder_dataset.load_dataset('C:\\Users\\maxim\\datasets\\1\\', recursive=False)
    
    # ts = date = reorder_dataset.get_date(dts[0])
    # te = date = reorder_dataset.get_date(dts[-1])

    its = [(reorder_dataset.get_date(i)-reorder_dataset.get_date(j)) for i, j in zip(dts[1:], dts[:2000]) if (reorder_dataset.get_date(i)-reorder_dataset.get_date(j))<1]
    av_its = 1/np.average(its)
    print(av_its)

    Y = []
    for d in dts:
        lab = autolib.get_label(d, flip=False)[0]
        date = reorder_dataset.get_date(d)
        Y.append(lab)

    Y = autolib.label_smoothing(Y, 5, 0) # to categorical

    pmap = pos_map(its=av_its, steer_coef=47)
    pos_list, minpos_list, vect_list, deg_list = pmap.get_pos(dts, Y[:2000])

    turns_segments, average = pmap.segment_track(pos_list, deg_list, look_back=60)
    plt.plot([i for i in range(len(vect_list))], np.array(vect_list)[:, 1], np.array(vect_list)[:, 0], linewidth=1)
    plt.plot(average, linewidth=1)
    plt.plot(its, linewidth=1)
    plt.show()

    iner, outer = pmap.boundaries(pos_list, r=0.8)
    diner = [1 for i in range(len(iner))]
    douter = [-1 for i in range(len(outer))]
    pmap.draw_points(pos_list+iner+outer, degs=deg_list+diner+douter)
    # cv2.imshow('pmap', pmap.pmap)
    # cv2.waitKey(0)

    pmap.draw_segments(turns_segments, min_max=iner+outer)
    cv2.imshow('pmap', pmap.pmap)
    cv2.waitKey(0)
