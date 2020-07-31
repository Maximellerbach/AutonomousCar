import sys
import time

import numpy as np

sys.path.append('C:\\Users\\maxim\\github\\AutonomousCar\\custom_modules\\')
sys.path.append('../custom_modules/')

import PID
import PIDController

if __name__ == "__main__":
    controller = PIDController.PID_controller()
    controller.update_target(1)
    # controller.init_ser()

    state = 0
    while(True):
        new_pwm = controller.update(state, time.time())
        if new_pwm != None:
            print(new_pwm)

        time.sleep(0.05)