import math
import random as rn

import cv2
import numpy as np
import scipy as sc
from constants import *
from scipy import interpolate
from scipy.spatial import ConvexHull

WIDTH = 250
HEIGHT = 250

# Boundaries for the numbers of points that will be randomly 
# generated to define the initial polygon used to build the track
MIN_POINTS = 20
MAX_POINTS = 30

SPLINE_POINTS = 100

# Margin between screen limits and any of the points that shape the
# initial polygon
MARGIN = 25
# minimum distance between points that form the track skeleton
MIN_DISTANCE = 50
# Maximum midpoint displacement for points placed after obtaining the initial polygon
MAX_DISPLACEMENT = 25
# Track difficulty
DIFFICULTY = 0.1
# min distance between two points that are part of thr track skeleton
DISTANCE_BETWEEN_POINTS = 10
# Maximum corner allowed angle
MAX_ANGLE = 90

class TrackGenerator():
    def __init__(self):
        self._track_points = []
        self._checkpoints = []

    # track generation methods
    def generate_track(self, debug=False, draw_checkpoints_in_track=False):
        # generate the track
        self._points = self.random_points()
        self._hull = ConvexHull(self._points)
        self._track_points = self.shape_track(self.get_track_points_from_hull(self._hull, self._points))
        self._track_points = self.smooth_track(self._track_points)

        self.draw_track_points(np.array(self._track_points))

    def save(self, points, x_offset=47.71, y_offset=0.6, z_offset=49.71787562201313, fact=1, save_file='C:\\Users\\maxim\\GITHUB\\sdsandbox\\sdsim\\Assets\\Resources\\track.txt'): # TODO:
        import os
        try:
            os.remove(save_file)
        except:
            pass

        points[-1] = points[0]

        points = np.array(points)*fact
        x_auto_off, z_auto_off = -points[0][0], -points[0][1]

        circuit_coords = open(save_file, 'w')
        for pt in points:
            circuit_coords.write(str(pt[0]+x_offset+x_auto_off)+','+str(y_offset)+','+str(pt[1]+z_offset+z_auto_off)+'\n')
        circuit_coords.close()
        
    def random_points(self, min=MIN_POINTS, max=MAX_POINTS, margin=MARGIN, min_distance=MIN_DISTANCE):
        point_count = rn.randrange(min, max+1, 1)
        points = []
        for i in range(point_count):
            x = rn.randrange(margin, WIDTH - margin + 1, 1)
            y = rn.randrange(margin, HEIGHT -margin + 1, 1)
            distances = list(
                filter(
                    lambda x: x < min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]
                )
            )
            if len(distances) == 0:
                points.append((x, y))
        return np.array(points)

    def get_track_points_from_hull(self, hull, points):
        # get the original points from the random 
        # set that will be used as the track starting shape
        return np.array([np.array(points[v]) for v in hull.vertices])

    def make_rand_vector(self, dims):
        vec = [rn.gauss(0, 1) for i in range(dims)]
        mag = sum(x**2 for x in vec) ** .5
        return [x/mag for x in vec]

    def shape_track(
        self,
        track_points,
        difficulty=DIFFICULTY,
        max_displacement=MAX_DISPLACEMENT,
        margin=MARGIN
    ):
        track_set = [[0,0] for i in range(len(track_points)*2)] 
        for i in range(len(track_points)):
            displacement = math.pow(rn.random(), difficulty) * max_displacement
            disp = [displacement * i for i in self.make_rand_vector(2)]
            track_set[i*2] = track_points[i]
            track_set[i*2 + 1][0] = (track_points[i][0] + track_points[(i+1)%len(track_points)][0]) / 2 + disp[0]
            track_set[i*2 + 1][1] = (track_points[i][1] + track_points[(i+1)%len(track_points)][1]) / 2 + disp[1]
        for i in range(3):
            track_set = self.fix_angles(track_set)
            track_set = self.push_points_apart(track_set)
        # push any point outside screen limits back again
        final_set = []
        for point in track_set:
            if point[0] < margin:
                point[0] = margin
            elif point[0] > (WIDTH - margin):
                point[0] = WIDTH - margin
            if point[1] < margin:
                point[1] = margin
            elif point[1] > HEIGHT - margin:
                point[1] = HEIGHT - margin
            final_set.append(point)
        return final_set

    def push_points_apart(self, points, distance=DISTANCE_BETWEEN_POINTS):
        # distance might need some tweaking
        distance2 = distance * distance 
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                p_distance =  math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
                if p_distance < distance:
                    dx = points[j][0] - points[i][0]  
                    dy = points[j][1] - points[i][1]  
                    dl = math.sqrt(dx*dx + dy*dy)  
                    dx /= dl
                    dy /= dl
                    dif = distance - dl
                    dx *= dif
                    dy *= dif
                    points[j][0] = points[j][0] + dx 
                    points[j][1] = points[j][1] + dy 
                    points[i][0] = points[i][0] - dx 
                    points[i][1] = points[i][1] - dy 
        return points

    def fix_angles(self, points, max_angle=MAX_ANGLE):
        for i in range(len(points)):
            if i > 0:
                prev_point = i - 1
            else:
                prev_point = len(points)-1
            next_point = (i+1) % len(points)
            px = points[i][0] - points[prev_point][0]
            py = points[i][1] - points[prev_point][1]
            pl = math.sqrt(px*px + py*py)
            px /= pl
            py /= pl
            nx = -(points[i][0] - points[next_point][0])
            ny = -(points[i][1] - points[next_point][1])
            nl = math.sqrt(nx*nx + ny*ny)
            nx /= nl
            ny /= nl  
            a = math.atan2(px * ny - py * nx, px * nx + py * ny)
            if (abs(math.degrees(a)) <= max_angle):
                continue
            diff = math.radians(max_angle * math.copysign(1,a)) - a
            c = math.cos(diff)
            s = math.sin(diff)
            new_x = (nx * c - ny * s) * nl
            new_y = (nx * s + ny * c) * nl
            points[next_point][0] = points[i][0] + new_x
            points[next_point][1] = points[i][1] + new_y
        return points

    def smooth_track(self, track_points):
        x = np.array([p[0] for p in track_points])
        y = np.array([p[1] for p in track_points])

        # append the starting x,y coordinates
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        tck, u = interpolate.splprep([x, y], s=0, per=True)

        # evaluate the spline fits for # points evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0, 1, SPLINE_POINTS), tck)
        return [(xi[i], yi[i]) for i in range(len(xi))]

    def draw_track_points(self, points):
        screen = np.zeros((HEIGHT, WIDTH, 1))
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(screen, [points], True, 1, thickness=2)
        cv2.imshow("track", screen)
        key = chr(cv2.waitKey(0))
        
        if key == "1":
            to_save_points = self.select_start_point(screen, self._track_points)
            self.save(to_save_points)

    def select_start_point(self, screen, points):
        start_index = 0
        key = "0"

        while(key != "1"):
            screen = np.zeros((HEIGHT, WIDTH, 1))
            for i, point in enumerate(points):
                cv2.circle(screen, (int(point[0]), int(point[1])), 10, 0.5)

            cv2.circle(screen, (int(points[start_index][0]), int(points[start_index][1])), 20, 1)

            cv2.imshow("track", screen)
            key = chr(cv2.waitKey(0))

            if key == "3": # plus
                start_index = int(start_index+1)
            elif key == "2": # minus
                start_index = int(start_index-1)

            start_index = start_index%len(points)

        points = np.concatenate((points[start_index:], points[:start_index]), axis=0)
        return points

    def rotate_circuit():
        return
            
if __name__ == "__main__":
    t = TrackGenerator()
    while(1):
        # try:
        #     t.generate_track()
        # except:
        #     print("unable to create track")
        t.generate_track()
