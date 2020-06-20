import math
import sys

import numpy as np

sys.path.append('custom_modules\\')



def plot_points(positions):
    import matplotlib.pyplot as plt

    positions = np.array(positions)
    X = positions[:, 1]
    Y = positions[:, 0]
    plt.scatter(X,Y)
    plt.axis('equal')
    plt.show()


def get_approx_distance(dt, speed):
    return dt*speed

def distance_needed_to_turn(angle, r):
    return 2*math.pi*r*(math.radians(angle)/math.pi)

def remaining_distance(dt, speed, d_remaining):
    d_remaining = d_remaining-(dt*speed)
    eta = d_remaining/speed
    return d_remaining, eta


def get_approx_radius(angle):
    # Read more about it: https://www.ntu.edu.sg/home/edwwang/confpapers/wdwicar01.pdf
    # https://www.youtube.com/watch?v=HqNdBiej23I
    
    L = 0.30

    angle = math.radians(angle)
    r_invert = math.sin(angle)/L
    r = 1/(r_invert+1E-8)

    return r


if __name__ == "__main__":
    
    r = get_approx_radius(38)
    d = distance_needed_to_turn(90, r)
    d_remaining, eta = remaining_distance(0.05, 5, d)
    print(r, d, d_remaining, eta)
