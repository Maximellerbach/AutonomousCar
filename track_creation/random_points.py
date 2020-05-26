import math
import random as rn
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import interpolate

from constants import *

class GameTrackGenerator():
    def __init__(self):
        self._track_points = []
        self._checkpoints = []

    # track generation methods
    def generate_track(self, debug=False, draw_checkpoints_in_track=False):
        # generate the track
        self._points = self.random_points()
        self._hull = ConvexHull(self._points)
        self._track_points = self.shape_track(self.get_track_points_from_hull(self._hull, self._points))
        self._f_points = self.smooth_track(self._track_points)

        self.draw_f_points(np.array(self._f_points))

    def save(self): # TODO:
        return
        
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
        vec = [rn.uniform(-1, 1) for i in range(dims)]
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
            track_set[i*2 + 1][0] = int((track_points[i][0] + track_points[(i+1)%len(track_points)][0]) / 2 + disp[0])
            track_set[i*2 + 1][1] = int((track_points[i][1] + track_points[(i+1)%len(track_points)][1]) / 2 + disp[1])
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
                    dx = points[j][0] - points[i][0];  
                    dy = points[j][1] - points[i][1];  
                    dl = math.sqrt(dx*dx + dy*dy);  
                    dx /= dl;  
                    dy /= dl;  
                    dif = distance - dl;  
                    dx *= dif;  
                    dy *= dif;  
                    points[j][0] = int(points[j][0] + dx);  
                    points[j][1] = int(points[j][1] + dy);  
                    points[i][0] = int(points[i][0] - dx);  
                    points[i][1] = int(points[i][1] - dy);  
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
            points[next_point][0] = int(points[i][0] + new_x)
            points[next_point][1] = int(points[i][1] + new_y)
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
        return [(int(xi[i]), int(yi[i])) for i in range(len(xi))]

    def draw_f_points(self, fpoints):
        plt.scatter(fpoints[:, 0], fpoints[:, 1])
        plt.show()
            
if __name__ == "__main__":
    t = GameTrackGenerator()
    for _ in range(10):
        t.generate_track()
        to_save = int(input('save ? '))
        if to_save:
            t.save()