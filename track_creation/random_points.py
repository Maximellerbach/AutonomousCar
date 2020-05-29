import math
import random as rn

import cv2
import numpy as np
import scipy as sc
from constants import *
from scipy import interpolate
from scipy.spatial import ConvexHull

# defining track width, height and z-axis height
WIDTH = 200
HEIGHT = 200
DEPTH = 4

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
MAX_DISPLACEMENT = 12.5

# Track difficulty
DIFFICULTY = 0.1
# min distance between two points that are part of thr track skeleton
DISTANCE_BETWEEN_POINTS = 25
# Maximum corner allowed angle
MAX_ANGLE = 90


class TrackGenerator():
    def __init__(self):
        self._track_points = []
        self._checkpoints = []

    # track generation methods
    def generate_track(self):
        # generate the track
        self._points = self.random_points()
        pts2D = np.array([self._points[:, 0], self._points[:, -1]]) 
        pts2D = np.transpose(pts2D, (1, 0))

        self._hull = ConvexHull(pts2D)
        self._track_points = self.shape_track(self.get_track_points_from_hull(self._hull, self._points))
        self._track_points = self.smooth_track(self._track_points)

        self.modify_track(self._track_points)

        
    def random_points(self, min=MIN_POINTS, max=MAX_POINTS, margin=MARGIN, min_distance=MIN_DISTANCE):
        def dist_2D(p, p2):
            x, z, y = p2
            return math.sqrt((p[0]-x)**2 + (p[1]-z)**2 + (p[2]-y)**2)
        point_count = rn.randrange(min, max+1, 1)
        points = []
        for i in range(point_count):
            x = rn.randrange(margin, WIDTH - margin + 1, 1)
            z = rn.uniform(0, 1)*DEPTH
            y = rn.randrange(margin, HEIGHT -margin + 1, 1)
            distances = list(
                filter(
                    lambda x: x < min_distance, [dist_2D(p, (x, z, y)) for p in points]
                )
            )
            if len(distances) == 0:
                points.append((x, z, y))
        return np.array(points)

    def get_track_points_from_hull(self, hull, points):
        # get the original points from the random 
        # set that will be used as the track starting shape
        pts = [np.array(points[v]) for v in hull.vertices]
        pts.append(pts[0])
        return np.array(pts)

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
        track_set = [[0,0,0] for i in range(len(track_points)*2)] 
        for i in range(len(track_points)):
            displacement = math.pow(rn.random(), difficulty) * max_displacement
            disp = [displacement * i for i in self.make_rand_vector(3)]
            track_set[i*2] = track_points[i]
            track_set[i*2 + 1][0] = (track_points[i][0] + track_points[(i+1)%len(track_points)][0]) / 2 + disp[0]
            track_set[i*2 + 1][-1] = (track_points[i][-1] + track_points[(i+1)%len(track_points)][-1]) / 2 + disp[1]
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
            if point[-1] < margin:
                point[-1] = margin
            elif point[-1] > HEIGHT - margin:
                point[-1] = HEIGHT - margin
            final_set.append(point)
        return final_set

    def push_points_apart(self, points, distance=DISTANCE_BETWEEN_POINTS):
        # distance might need some tweaking
        distance2 = distance * distance 
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                p_distance =  math.sqrt((points[i][0]-points[j][0])**2 + (points[i][-1]-points[j][-1])**2)
                if p_distance < distance:
                    dx = points[j][0] - points[i][0]  
                    dy = points[j][-1] - points[i][-1]  
                    dl = math.sqrt(dx*dx + dy*dy)  
                    dx /= dl
                    dy /= dl
                    dif = distance - dl
                    dx *= dif
                    dy *= dif
                    points[j][0] = points[j][0] + dx 
                    points[j][-1] = points[j][-1] + dy 
                    points[i][0] = points[i][0] - dx 
                    points[i][-1] = points[i][-1] - dy 
        return points

    def fix_angles(self, points, max_angle=MAX_ANGLE):
        for i in range(len(points)):
            if i > 0:
                prev_point = i - 1
            else:
                prev_point = len(points)-1
            next_point = (i+1) % len(points)
            px = points[i][0] - points[prev_point][0]
            py = points[i][-1] - points[prev_point][-1]
            pl = math.sqrt(px*px + py*py)
            px /= pl
            py /= pl
            nx = -(points[i][0] - points[next_point][0])
            ny = -(points[i][-1] - points[next_point][-1])
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
            points[next_point][-1] = points[i][-1] + new_y
        return points

    def smooth_track(self, track_points):
        x = np.array([p[0] for p in track_points])
        z = np.array([p[1] for p in track_points])
        y = np.array([p[-1] for p in track_points])


        # append the starting x,y coordinates
        x = np.r_[x, x[0]]
        z = np.r_[z, z[0]]
        y = np.r_[y, y[0]]

        # fit splines to x=f(u) and y=g(u), treating both as periodic.
        tck, u = interpolate.splprep([x, z, y], s=3, per=True)

        # evaluate the spline fits for # points evenly spaced distance values
        xi, zi, yi = interpolate.splev(np.linspace(0, 1, SPLINE_POINTS), tck)
        return [(xi[i], zi[i], yi[i]) for i in range(len(xi))]

    def make_z(self, points):
        return

    def modify_track(self, points):
        screen = np.zeros((HEIGHT, WIDTH, 1))
        points = np.array(points, dtype=np.int32)
        to_draw = np.array([(i[0], i[-1]) for i in points])
        to_draw = to_draw.reshape((-1,1,2))
        cv2.polylines(screen, [to_draw], True, 1, thickness=2)
        cv2.imshow("track", screen)
        key = chr(cv2.waitKey(0))
        
        if key == "a": # saving process
            to_save_points, start_index = self.select_start_point(self._track_points)
            to_save_points = self.rotate_circuit(to_save_points, start_index)
            to_save_points = self.vertical_flip_points(to_save_points)
            self.save(to_save_points)
    
    def save(self, pts, x_offset=47.71, y_offset=0.6, z_offset=49.71787562201313, fact=1, save_file='C:\\Users\\maxim\\GITHUB\\sdsandbox\\sdsim\\Assets\\Resources\\track.txt'): # TODO:
        import os
        try:
            os.remove(save_file)
        except:
            pass
        
        pts.append(pts[0]) # close the loop

        pts = np.array(pts)*fact
        x_auto_off, z_auto_off, y_auto_off = -pts[0]
        offset_array = np.array((x_auto_off, z_auto_off, y_auto_off))
        for pt in pts:
            pt += offset_array
            if pt[1] < 0:
                pt[1] = 0

        circuit_coords = open(save_file, 'w')
        for pt in pts:
            circuit_coords.write(str(pt[0]+x_offset)+','+str(pt[1]+y_offset)+','+str(pt[-1]+z_offset)+'\n')
        circuit_coords.close()

    def select_start_point(self, points):
        start_index = 0
        key = ""

        while(key != "a"):
            screen = np.zeros((HEIGHT, WIDTH, 3))
            d_max = max(np.array(points)[:, 1])
            for i, point in enumerate(points):
                cv2.circle(screen, (int(point[0]), int(point[-1])), 8, (1, point[1]/d_max, 0))

            cv2.circle(screen, (int(points[start_index][0]), int(points[start_index][-1])), 20, (1, 1, 1))

            cv2.imshow("track", screen)
            key = chr(cv2.waitKey(0))

            if key == "p": # plus
                start_index = int(start_index+1)
            elif key == "m": # minus
                start_index = int(start_index-1)

            start_index = start_index%len(points)

        not_empty = [i for i in (points[start_index:], points[:start_index]) if len(i)!=0]
        points = list(np.concatenate(not_empty, axis=0))
        del points[-start_index]
        return points, start_index # start_index is now 0, this is more of an offset value

    def rotate_circuit(self, points, start_index):
        def rotate(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            tmp_angle = math.radians(angle)
            ox, oy = origin
            px, pz, py = point

            qx = ox + math.cos(tmp_angle) * (px - ox) - math.sin(tmp_angle) * (py - oy)
            qy = oy + math.sin(tmp_angle) * (px - ox) + math.cos(tmp_angle) * (py - oy)
            return qx, pz, qy

        angle = 0
        key = ""
        mid_point = (WIDTH//2, HEIGHT//2)

        new_points = points
        while(key != "a"):
            screen = np.zeros((HEIGHT, WIDTH, 3))
            d_max = max(np.array(points)[:, 1])
            for i, point in enumerate(new_points):
                cv2.circle(screen, (int(point[0]), int(point[-1])), 8, (1, point[1]/d_max, 0))

            st = new_points[0]
            cv2.line(screen, (int(st[0]-10), int(st[-1])), (int(st[0]+10), int(st[-1])), (1,1,1), thickness=2)
            cv2.imshow("track", screen)
            key = chr(cv2.waitKey(0))

            if key == "p": # plus
                angle = int(angle+1)
            elif key == "m": # minus
                angle = int(angle-1)

            angle = angle%360
            new_points = [rotate(mid_point, pt, angle) for pt in points]

        return new_points

    def vertical_flip_points(self, points):
        points = np.array(points)
        points[:, -1] = -points[:, -1]+HEIGHT
        return list(points)

            
if __name__ == "__main__":
    t = TrackGenerator()
    while(1):
        # try:
        #     t.generate_track()
        # except:
        #     print("unable to create track")
        t.generate_track()
