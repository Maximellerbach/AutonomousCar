import time

import math

from . import PID


class PIDController:
    def __init__(self, kp=1, ki=1, kd=1):
        self.pid = PID.PID(kp, ki, kd)
        self.pid.SetPoint = 0  # neutral point

        self.last_received = time.time()
        self.current_speed = 0
        self.pwm = 0

        self.update_target(0)

    def update(self, current_speed, time_received):
        if time_received != self.last_received:
            self.pid.update(current_speed, current_time=time_received)
            self.last_received = time_received
            self.pwm = self.pid.output
            print("throttle", self.pwm)

            # # basic filter
            # if new_pwm < self.high_th and new_pwm > -self.low_th:
            #     self.pwm = new_pwm + self.low_th
            # elif new_pwm <= -self.low_th:
            #     self.pwm -= 5
            # else:
            #     self.pwm = self.high_th

            self.pwm = max(min(self.pwm, 127), 0)

            self.last_received = time_received
        return self.pwm

    def update_target(self, new_target):
        self.pid.SetPoint = new_target
