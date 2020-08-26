import math

import cv2
import numpy as np
from tqdm import tqdm

from custom_modules.datasets import dataset_json
from custom_modules.vis import plot


class TrackEstimation():
    def __init__(self, size=(512, 512), steer_coef=30):
        self.pmap = np.zeros((size[0], size[1], 3))
        self.steer_coef = steer_coef
        self.segment_map = []
        self.angle = 0

    def clear_pmap(self):
        '''
        Function to clear the pmap
        takes no args, returns nothing
        '''
        self.pmap = np.zeros_like(self.pmap)

    def point_distance(self, pt1, pt2):
        '''
        function to process distance between 2 points
        '''
        d = math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
        return d

    def max_min(self, segments):
        '''
        function to return maximum and minimum of points in a "segments" array of shape (n, 2, 4)
        '''
        segments_points = segments[:, :, :2]
        maxp = np.max(segments_points)
        minp = np.min(segments_points)

        return maxp, minp

    def map_pt(self, p, maxp, minp, is_integer=True, size=(512, 512)):
        '''
        Function to remap a point with a given max and min and a desired final size, precise is_integer=False if you want your point to be (float, float)
        (p, maxp, minp, is_integer, size) -> (x, y)
        (Iterable of len==2, float, float, bool, tuple (h,w)) -> (float/int, float/int)

        '''
        h, w = size

        x = ((p[0]-minp)/(maxp-minp))*w
        y = ((p[1]-minp)/(maxp-minp))*h
        if is_integer:
            x = int(x)
            y = int(y)

        return (x, y)

    def normalize_pts(self, pts_segm, maxp, minp):
        '''
        function to normalize two segment points according to their position and turn angle
        '''
        pt1, pt2 = pts_segm
        p1 = self.map_pt(pt1[:2], maxp, minp, is_integer=False, size=(1, 1))
        p2 = self.map_pt(pt2[:2], maxp, minp, is_integer=False, size=(1, 1))
        av = np.average([p1, p2], axis=-1)
        # multiply positive coords with sign (- or +) to separate left turns from right turns -> output coords [-1; 1]
        return av * pt1[-1]

    def get_pos(self, Y, time_series, speed_series):
        '''
        Function to calculate position from image label,
        here I'm making a few assumptions like speed = cte, delta_time = cte and the trajectory is not arced
        It's a rough estimation to get something concrete to visualize and turning angle
        (Y, time_series, speed_series, speed) -> (pos_list, lightpos_list, vect_list, deg_list)
        ([], [floats], [floats], float) _> ([(x, y)], [(x, y)], [(dx, dy)], [floats])
        '''
        vect_list = []
        pos_list = []
        lightpos_list = []
        deg_list = []
        pos = [0, 0]
        angle = 0

        for it, st in enumerate(Y):
            st = -st  # transform [-1, 1] to [1, -1]

            # TODO current steering angle estimation # https://stackoverflow.com/questions/25895222/estimated-position-of-vector-after-time-angle-and-speed

            degree_angle = math.pi * st * self.steer_coef * \
                (speed_series[it]*time_series[it])
            rad = math.radians(degree_angle)

            angle += rad
            angle = angle % (2*math.pi)

            x = math.sin(angle)*speed_series[it]*time_series[it]
            y = math.cos(angle)*speed_series[it]*time_series[it]

            pos[0] += x
            pos[1] += y

            pos_list.append((pos[0], pos[1]))
            vect_list.append((x, y))
            deg_list.append(rad)

            if degree_angle != 0. or it == 0:
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

            way = radius if x2-x1 > 0 else -radius

            if (x2-x1) != 0:
                a = (y2-y1)/(x2-x1)

                angle = math.pi/2+math.atan(a)
                wi = math.cos(angle)*offset*way
                hi = math.sin(angle)*offset*way

                iner.append((x1+wi, y1+hi))
                outer.append((x1-wi, y1-hi))

            else:
                iner.append((x1+offset*way, y1))
                outer.append((x1-offset*way, y1))

        return iner, outer

    def segment_track(self, pos_list, degs, th=0.005, look_back=30):
        '''
        Function to detect turns in the trajectory according to the average rotation of the car (look_back is the number of iterations averaged) by applying a threshold.
        After that, storing the [x,y,it,sign] of the start and the end of the turn, returning this array of shape (n, 2, 4) and averaged rotation
        '''

        turning = False
        way = 0
        average = [0]*len(degs)
        thresholded = []
        turn = []

        for it in range(look_back//2, len(degs)-look_back//2):
            average[it] = np.average(degs[it-look_back//2:it+look_back//2])

            if average[it] >= th:
                if turning is False:
                    way = 1
                    x, y = pos_list[it]
                    turn.append((x, y, it, way))
                    turning = True

            elif average[it] <= -th:
                if turning is False:
                    way = -1
                    x, y = pos_list[it]
                    turn.append((x, y, it, way))
                    turning = True

            elif turning:
                x, y = pos_list[it]
                turn.append((x, y, it, way))
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
        maxp, minp = self.max_min(segments)
        # set of turns segments, len(segments) should be greater then the number of turns in a lap
        X = [self.normalize_pts(pts, maxp, minp) for pts in segments]
        # 2 -> number of turns and 2 of distance (max distance for [-1; 1])
        best = [2, 2]
        for n_turns in tqdm(range(2, len(X)+1)):
            pattern = [i % n_turns for i in range(len(X))]
            pattern_points = [[] for i in range(n_turns)]
            distance_turns = [[] for i in range(n_turns)]
            loss_list = [0]*n_turns
            loss_n_list = [0]*n_turns

            for i in range(len(pattern)):
                pattern_points[pattern[i]].append(X[i])

            for it, pts in enumerate(pattern_points):
                for i in range(len(pts)):
                    for pt in pts:
                        d = self.point_distance(pt, pts[i])
                        distance_turns[it].append(d)
                        loss_list[it] += d
                        loss_n_list[it] += 1

            loss = [i/j for i, j in zip(loss_list, loss_n_list)]
            if loss != [0]*len(loss):  # avoid best score to be n_turn (1 match per turn)
                av = np.average([i for i in loss if i != 0])
                if av < best[1]:
                    best = [n_turns, av]

        return best

    def evaluate_n_turns(self, segments, n_turns):
        '''
        evaluate loss for n_turns
        '''

        maxp, minp = self.max_min(segments)
        # set of turns segments, len(segments) should be greater then the number of turns in a lap
        X = [self.normalize_pts(pts, maxp, minp) for pts in segments]
        pattern = [i % n_turns for i in range(len(X))]
        pattern_points = [[] for i in range(n_turns)]
        distance_turns = [[] for i in range(n_turns)]
        loss_list = [0]*n_turns
        loss_n_list = [0]*n_turns

        for i in range(len(pattern)):
            pattern_points[pattern[i]].append(X[i])

        for it, pts in enumerate(pattern_points):
            for i in range(len(pts)):
                for pt in pts:
                    d = self.point_distance(pt, pts[i])
                    distance_turns[it].append(d)
                    loss_list[it] += d
                    loss_n_list[it] += 1

        loss = [i/j for i, j in zip(loss_list, loss_n_list)]
        if loss != [0]*len(loss):  # avoid best score to be n_turn (1 match per turn)
            av = np.average([i for i in loss if i != 0])

        return av

    def match_segments(self, segments, n_turns=0):
        '''
        returns pattern, n_turn and accuracy. if n_turns is not precised or < 2, process ideal n_turns with optimize_n_segments
        '''
        if n_turns < 2:
            n_turns, loss = self.optimize_n_segments(segments)
        else:
            loss = self.evaluate_n_turns(segments, n_turns)
        matchs = [i % n_turns for i in range(len(segments))]
        return matchs, n_turns, 1-loss

    def create_colors(self, classes):
        return [np.random.random(3) for _ in range(classes)]

    def draw_points(self, pos_list, degs, colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], thickness=1):
        # there is 3 states: forward, left and right, thus 3 colors
        assert len(colors) == 3

        maxp = np.max(pos_list)
        minp = np.min(pos_list)

        h, w, _ = self.pmap.shape
        for pos, deg in zip(pos_list, degs):
            rpx = ((pos[0]-minp)/(maxp-minp))*w
            rpy = ((pos[1]-minp)/(maxp-minp))*h

            if deg > 0:
                color = colors[0]*deg
            elif deg < 0:
                color = colors[2]*deg
            else:
                color = colors[1]

            cv2.circle(self.pmap, (int(rpx), int(rpy)),
                       1, color, thickness=thickness)

    def draw_circuit(self, pos_list, degs, colors=np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))), thickness=1):
        # there is 3 states: forward, left and right, thus 3 colors
        assert len(colors) == 3
        inner, outer = Estimator.boundaries(pos_list, radius=5)

        to_draw = pos_list + inner + outer
        degs = degs[:-1] + [-1]*len(inner) + [1]*len(outer)

        maxp = np.max(to_draw)
        minp = np.min(to_draw)

        h, w, _ = self.pmap.shape
        for pos, deg in zip(to_draw, degs):
            rpx = ((pos[0]-minp)/(maxp-minp))*w
            rpy = ((pos[1]-minp)/(maxp-minp))*h

            if deg > 0:
                color = colors[0]*deg
            elif deg < 0:
                color = colors[2]*deg
            else:
                color = colors[1]

            cv2.circle(self.pmap, (int(rpx), int(rpy)),
                       1, color, thickness=thickness)

    def draw_segments(self, segments, matches=[], min_max=[]):
        segments = np.array(segments)
        assert (len(segments.shape) == 3)  # There is no points in there !
        segments_points = segments[:, :, :2]

        if min_max == []:
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
                if way > 0:
                    color = (1, 0, 0)
                elif way < 0:
                    color = (0, 0, 1)

            cv2.line(self.pmap, p1, p2, color, thickness=3)


if __name__ == "__main__":
    # 'C:\\Users\\maxim\\recorded_imgs\\0_0_1587729884.301688\\' # 'C:\\Users\\maxim\\datasets\\1 ironcar driving\\'
    Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
    direction_comp = Dataset.get_component("direction")
    direction_comp.offset = -7
    direction_comp.scale = 1/4

    paths = Dataset.load_dos_sorted(
        'C:\\Users\\maxim\\random_data\\json_dataset\\12 sim circuit 2 new\\')
    sequence_to_study = (0, 1000)
    paths = paths[sequence_to_study[0]:sequence_to_study[1]]

    annotations = np.array([Dataset.load_annotation(path) for path in paths])

    directions = annotations[:, 0][:-1]
    speeds = annotations[:, 1]
    dates = annotations[:, -1]
    delta_times = dates[1:]-dates[:-1]

    Estimator = TrackEstimation(steer_coef=14)

    pos_list, lightpos_list, vect_list, deg_list = Estimator.get_pos(
        directions, time_series=delta_times, speed_series=speeds)

    plot.plot_time_series(np.array(vect_list)[:, 0], np.array(vect_list)[:, 1])

    iner_bound_list, outer_bound_list = Estimator.boundaries(
        pos_list, radius=5)
    diner = [1 for i in range(len(iner_bound_list))]
    douter = [-1 for i in range(len(outer_bound_list))]

    turns_segments, average = Estimator.segment_track(
        pos_list, directions, th=0.5, look_back=10)

    matchs, n_turns, accuracy = Estimator.match_segments(turns_segments)
    print(matchs, '| number of turns in a lap: ',
          n_turns, '| accuracy: ', accuracy)

    Estimator.draw_segments(turns_segments, matches=matchs,
                            min_max=iner_bound_list+outer_bound_list)

    Estimator.draw_circuit(pos_list, directions, colors=np.array(
        ((0.75, 0, 0), (1, 1, 1), (0, 0, 0.75))))

    cv2.imshow('pmap', Estimator.pmap)
    cv2.waitKey(0)
