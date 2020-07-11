import math
import sys
import os
import time

import numpy as np

sys.path.append('../custom_modules/')
import SerialCommand
import PID

def start_serial(port="/dev/ttyUSB0"):
    os.system('sudo chmod 0666 {}'.format(port))
    ser = SerialCommand.control(port)
    return ser


if __name__ == "__main__":
    ser = start_serial()

    kp = 1
    ki = 1
    kd = 0

    pid = PID.PID(kp, ki, kd)
    current_speed = ser.GetCurrentSpeed()

    ser.ChangeMotorA(1)

    to_target = 1
    pid.SetPoint = to_target

    last_received = time.time()
    while(True):

        current_speed = ser.GetCurrentSpeed()
        time_received = ser.GetTimeLastReceived()

        if time_received != last_received:
            pid.update(current_speed, current_time=last_received)
            new_pwm = int(pid.output*255)
            last_received = time_received

            ser.ChangePWM(new_pwm)
            print(new_pwm)

    


