import math
import sys
import time

import numpy as np

sys.path.append('../custom_modules/')
import SerialCommand

def start_serial(port="/dev/ttyUSB0"):
    ser = SerialCommand.control(port)
    return ser

def update_throttle(speed, dt, prev_throttle, prev_speed, target_speed=3, th_norm=255, changing_th_ratio=0.05):

    normalized_speed = speed/target_speed
    prev_normalized_speed = speed/target_speed
    delta_speed = normalized_speed-prev_normalized_speed # delta speed calculated from last mesures
    normalized_throttle = prev_throttle/th_norm    

    acceleration_rate = delta_speed/dt


    


    return 


if __name__ == "__main__":
    ser = start_serial()

    current_speed = ser.GetCurrentSpeed()
    



