import math
import sys
import time

import numpy as np

sys.path.append('../custom_modules/')
import SerialCommand

dico = [10,8,6,4,2]

def start_serial(port="/dev/ttyUSB0"):
    ser = SerialCommand.control(port)
    return ser

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

def remaining_distance(distance, d_remaining):
    d_remaining = d_remaining-distance
    return d_remaining


def get_approx_radius(angle):
    # Read more about it: https://www.ntu.edu.sg/home/edwwang/confpapers/wdwicar01.pdf
    # https://www.youtube.com/watch?v=HqNdBiej23I
    
    L = 0.30

    angle = math.radians(angle)
    r_invert = math.sin(angle)/L
    r = 1/(r_invert+1E-8)

    return r

def rotatecar(ser, angle, way, max_angle=40, wheel_length=0.32, orientation=1):
    if way == 2:
        mult = -1
    else:
        mult = 1
        
    r = get_approx_radius(max_angle)
    d_remaining = distance_needed_to_turn(angle, r)*mult
    remaining = d_remaining

    time.sleep(1) # wait for the get turn thread to start

    start_turns = -ser.GetTurns()
    start_time = ser.GetTimeLastReceived()
    prev_turns = start_turns

    print(start_turns)

    if orientation == 1:
        direction = 0
    else:
        direction = -1

    ser.ChangeDirection(dico[direction])
    ser.ChangeMotorA(way)
    ser.ChangePWM(85)
    
    it = 0
    while((remaining*mult)>0): # stop 10cm before (inertia)
        in_progress_turns = -ser.GetTurns()
        in_progress_time = ser.GetTimeLastReceived()
        # print(in_progress_turns, in_progress_time)

        if in_progress_turns != prev_turns:
            delta_turns = in_progress_turns-start_turns #turns are actually counted downwards when going forward, reversing it
            dt = start_time-in_progress_time
            delta_distance = (wheel_length*(delta_turns/6))
            # if delta_distance/dt < 10: # set a threshold of 10m/s
            remaining = remaining_distance(delta_distance, d_remaining)
                
            prev_turns = in_progress_turns
            it += 1

            print(in_progress_turns, delta_distance, remaining, it)
        
        
    ser.ChangePWM(0)
    ser.ChangeDirection(dico[2])
    ser.ChangeMotorA(0)

if __name__ == "__main__":
        
    def getParams(argv):
        for it, arg in enumerate(argv):
            if arg in ("-a", "--angle"):
                angle = int(argv[it+1].strip())
            elif arg in ("-w", "--way"):
                way = int(argv[it+1].strip())
            elif arg in ("-o", "--orientation"):
                orientation = int(argv[it+1].strip())
                
        return angle, way, orientation

    # r = get_approx_radius(38)
    # d = distance_needed_to_turn(90, r)
    # d_remaining = remaining_distance(0.5, d)
    # print(r d, d_remaining)

    angle, way, orientation = getParams(sys.argv[1:])

    ser = start_serial()
    rotatecar(ser, angle, way, orientation=orientation)

    time.sleep(1)
    ser.stop()
    print("got here")