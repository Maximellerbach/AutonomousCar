from .. import serial_command
import math
import time

dico = [10, 8, 6, 4, 2]


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


def rotatecar(ser, angle, way, max_angle=40, wheel_length=0.25, orientation=1):
    if way == 2:
        mult = -1
    else:
        mult = 1

    r = get_approx_radius(max_angle)
    d_remaining = distance_needed_to_turn(angle, r)*mult
    remaining = d_remaining

    time.sleep(1)  # wait for the get turn thread to start

    start_turns = ser.GetTurns()
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
    # refactor This
    while((remaining*mult) > 0):  # stop 10cm before (inertia)
        in_progress_turns = ser.GetTurns()
        in_progress_time = ser.GetTimeLastReceived()
        # print(in_progress_turns, in_progress_time)

        if in_progress_turns != prev_turns:
            delta_turns = in_progress_turns-start_turns
            dt = start_time-in_progress_time
            delta_distance = (wheel_length*(delta_turns/6))
            
            remaining = remaining_distance(delta_distance, d_remaining)

            prev_turns = in_progress_turns
            it += 1

            print(in_progress_turns, delta_distance, remaining, dt, it)

    ser.ChangePWM(0)
    ser.ChangeDirection(dico[2])
    ser.ChangeMotorA(0)
