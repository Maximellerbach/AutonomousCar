import autolib
import math
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from tqdm import tqdm

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
        '''
        Function to clear the pmap
        takes no args, returns nothing
        '''
        self.pmap = np.zeros_like(self.pmap)

    def map_pt(self, p, maxp, minp, integer=True, size=(512, 512)):
        '''
        Function to remap a point with a given max and min and a desired final size, precise integer=False if you want your point to be (float, float)
        (p, maxp, minp, integer, size) -> (x, y)
        (Iterable of len==2, float, float, bool, tuple (h,w)) -> (float/int, float/int)

        '''
        h, w = size

        x = ((p[0]-minp)/(maxp-minp))*w
        y = ((p[1]-minp)/(maxp-minp))*h
        if integer == True:
            x = int(x)
            y = int(y)

        return (x, y)

    def get_pos(self, Y, speed=1):
        '''
        Function to calculate position from image label,
        here I'm making a few assumptions like speed = cte, delta_time = cte and the trajectory is not arced
        It's a rough estimation to get something concrete to visualize and turning angle
        (Y, speed) -> (pos_list, lightpos_list, vect_list, deg_list)
        ([], float) _> ([(x, y)], [(x, y)], [(dx, dy)], [floats])
        '''
        vect_list = []
        pos_list = []
        lightpos_list = []
        deg_list = []
        pos = [0, 0]
        angle = 0

        for it, ny in enumerate(Y):
            average = 0
            coef = [1, 0.5, 0, -0.5, -1]

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
            deg_list.append(rad)

            if degree_angle!=0. or it==0:
                lightpos_list.append((pos[0], pos[1], it))
            

        return pos_list, lightpos_list, vect_list, deg_list


    def boundaries(self, pos_list, radius=1):
        '''
        Function to calculate outer and iner boundaries of the circuit from a trajectory and a radius (distance between center and border)
        (pos_list, radius) -> (iner, outer)
        ([], float) _> ([(x, y)], [(x, y)])
        '''
        iner = []
        outer = []

        offset = 1
        for p1, p2 in zip(pos_list[:-1], pos_list[1:]):
            x1, y1 = p1[:2]
            x2, y2 = p2[:2]

            way = radius if x2-x1>0 else -radius

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
        '''
        Function to detect turns in the trajectory according to the average rotation of the car (look_back is the number of iterations averaged) by applying a threshold.
        After that, storing the [x,y,it,sign] of the start and the end of the turn, returning this array of shape (n, 2, 4) and averaged rotation
        '''
        
        turning = False
        way = 0
        average = [0]*len(deg_list)
        thresholded = []
        turn = []

        for it in range(look_back, len(deg_list)):
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

        return np.array(thresholded), np.array(average)

    def optimize_n_segments(self, segments):
        '''
        Function to find the ideal number of turns in a lap using distance between points
        ([[x, y, it, sign]*2]) -> (n_turns)
        ([[float, float, int, int]*2]) -> (int)
        '''

        assert (len(segments.shape)==3) # There is no points in there !
        segments_points = segments[:, :, :2]
        maxp = np.max(segments_points)
        minp = np.min(segments_points)

        def point_distance(pt1, pt2):
            d = math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
            return d

        def transform(pts):
            pt1, pt2 = pts
            p1 = self.map_pt(pt1[:2], maxp, minp, integer=False, size=(1, 1))
            p2 = self.map_pt(pt2[:2], maxp, minp, integer=False, size=(1, 1))
            av = np.average([p1, p2], axis=-1)
            return av * pt1[-1] # multiply positive coords with sign (- or +) to separate left turns from right turns -> output coords [-1; 1]

        X = [transform(pts) for pts in segments] # set of turns segments, len(segments) should be greater then the number of turns in a lap
        best = [2, 2] # 2 -> number of turns and 2 of distance (max distance for [-1; 1])
        for n_turns in tqdm(range(2, len(X)+1)):
            pattern = [i%n_turns for i in range(len(X))]
            pattern_points = [[] for i in range(n_turns)]
            distance_turns = [[] for i in range(n_turns)]
            loss_list = [0]*n_turns
            loss_n_list = [0]*n_turns

            for i in range(len(pattern)):
                pattern_points[pattern[i]].append(X[i])

            for it, pts in enumerate(pattern_points):
                for i in range(len(pts)):
                    for pt in pts:
                        d = point_distance(pt, pts[i])
                        distance_turns[it].append(d)
                        loss_list[it] += d
                        loss_n_list[it] += 1

            loss = [i/j for i, j in zip(loss_list, loss_n_list)]
            if loss != [0]*len(loss):# avoid best score to be n_turn (1 match per turn)
                av = np.average([i for i in loss if i!= 0])
                if av<best[1]: 
                    best = [n_turns, av]

        return best

    def evaluate_n_turns(self, segments, n_turns):
        '''
        evaluate loss for n_turns
        '''
        assert (len(segments.shape)==3) # There is no points in there !
        segments_points = segments[:, :, :2]
        maxp = np.max(segments_points)
        minp = np.min(segments_points)

        def point_distance(pt1, pt2):
            d = math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
            return d

        def transform(pts):
            pt1, pt2 = pts
            p1 = self.map_pt(pt1[:2], maxp, minp, integer=False, size=(1, 1))
            p2 = self.map_pt(pt2[:2], maxp, minp, integer=False, size=(1, 1))
            av = np.average([p1, p2], axis=-1)
            return av * pt1[-1] # multiply positive coords with sign (- or +) to separate left turns from right turns -> output coords [-1; 1]

        X = [transform(pts) for pts in segments] # set of turns segments, len(segments) should be greater then the number of turns in a lap
        pattern = [i%n_turns for i in range(len(X))]
        pattern_points = [[] for i in range(n_turns)]
        distance_turns = [[] for i in range(n_turns)]
        loss_list = [0]*n_turns
        loss_n_list = [0]*n_turns

        for i in range(len(pattern)):
            pattern_points[pattern[i]].append(X[i])

        for it, pts in enumerate(pattern_points):
            for i in range(len(pts)):
                for pt in pts:
                    d = point_distance(pt, pts[i])
                    distance_turns[it].append(d)
                    loss_list[it] += d
                    loss_n_list[it] += 1

        loss = [i/j for i, j in zip(loss_list, loss_n_list)]
        if loss != [0]*len(loss):# avoid best score to be n_turn (1 match per turn)
            av = np.average([i for i in loss if i!= 0])

        return av

    def match_segments(self, segments, n_turns=0):
        '''
        returns pattern, n_turn and accuracy. if n_turns is not precised or < 2, process ideal n_turns with optimize_n_segments
        '''
        if n_turns < 2:
            n_turns, loss = self.optimize_n_segments(segments)
        else:
            loss = self.evaluate_n_turns(segments, n_turns)
        matchs = [i%n_turns for i in range(len(segments))]
        return matchs, n_turns, 1-loss
    
    def create_colors(self, classes):
        return [np.random.random(3) for _ in range(classes)]


    def draw_points(self, pos_list, degs=[], colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
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
                    color = colors[0]
                    th = 1
                elif degs[it]<0:
                    color = colors[2]
                    th = 1
                else:
                    color = colors[1]
                    th = 1
                    
            except:
                color = (0, 1, 0)
                th = 1


            cv2.circle(self.pmap, (int(rpx), int(rpy)), 1, color, thickness=th)
    
    def draw_segments(self, segments, matches=[], min_max=[]):
        segments = np.array(segments)
        assert (len(segments.shape)==3) # There is no points in there !
        segments_points = segments[:, :, :2]

        if min_max==[]:
            maxp = np.max(segments_points)
            minp = np.min(segments_points)
        else:
            maxp = np.max(min_max)
            minp = np.min(min_max)

        if matches != []:
            colors = self.create_colors(max(matches)+1)

        for i, segm in enumerate(segments):
            p1 = self.map_pt(segm[0], maxp, minp, size=self.pmap.shape[:2])
            p2 = self.map_pt(segm[1], maxp, minp, size=self.pmap.shape[:2])
            way = segm[0][-1]

            if matches != []:
                match_number = matches[i]
                color = colors[match_number]
                
            else:
                if way>0:
                    color = (1, 0, 0)
                elif way<0:
                    color = (0, 0, 1)

            cv2.line(self.pmap, p1, p2, color, thickness=3)

if __name__ == "__main__":
    dts, datalen = reorder_dataset.load_dataset('C:\\Users\\maxim\\datasets\\1\\', recursive=False)
    sequence_to_study = (0, 2500)

    dates = [reorder_dataset.get_date(i) for i in dts]
    its = [i-j for i, j in zip(dates[sequence_to_study[0]+1:sequence_to_study[1]+1], dates[sequence_to_study[0]:sequence_to_study[1]]) if i-j<0.1] # remove images where dt >= 0.1
    av_its = 1/np.average(its)
    print("average img/sec:", av_its, "| images removed:", (sequence_to_study[1]-sequence_to_study[0])-len(its))

    Y = []
    for d in dts:
        lab = autolib.get_label(d, flip=False)[0]
        date = reorder_dataset.get_date(d)
        Y.append(lab)

    Y = autolib.label_smoothing(Y, 5, 0) # to categorical

    pmap = pos_map(its=av_its, steer_coef=47)
    pos_list, lightpos_list, vect_list, deg_list = pmap.get_pos(Y[sequence_to_study[0]:sequence_to_study[1]], speed=1)

    turns_segments, average = pmap.segment_track(pos_list, deg_list, th=0.007, look_back=60)
    
    # plt.plot([i for i in range(len(vect_list))], np.array(vect_list)[:, 1], np.array(vect_list)[:, 0], linewidth=1)
    # plt.plot(average, linewidth=1)
    # plt.plot(its, linewidth=1) # useless unless you want to see consistency of NN/image saves 
    # plt.show()

    matchs, n_turns, accuracy = pmap.match_segments(turns_segments, n=i)
    print(matchs, '| number of turns in a lap: ', n_turns, '| accuracy: ', accuracy)

    iner_list, outer_list = pmap.boundaries(pos_list, radius=0.8)
    diner = [1 for i in range(len(iner_list))]
    douter = [-1 for i in range(len(outer_list))]

    pmap.draw_points(pos_list+iner_list+outer_list, degs=deg_list+diner+douter, colors=[(0.75, 0, 0), (1, 1, 1), (0, 0, 0.75)])
    pmap.draw_segments(turns_segments, matches=matchs, min_max=iner_list+outer_list)

    cv2.imshow('pmap', pmap.pmap)
    cv2.waitKey(0)
