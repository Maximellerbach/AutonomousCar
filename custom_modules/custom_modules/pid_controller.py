import time
from . import PID


class PIDController():
    def __init__(self, kp=1, ki=1, kd=1, high_th=127, low_th=30):
        self.pid = PID.PID(kp, ki, kd)
        self.ser = None

        self.high_th = high_th
        self.low_th = low_th

        self.last_received = time.time()
        self.current_speed = 0
        self.pwm = 0

        self.update_target(0)

    def init_ser(self, ser):
        self.ser = ser
        time.sleep(1)  # wait for the sensor data to come in

        self.ser.ChangeMotorA(1)  # set motor A to forward
        self.ser.ChangePWM(0)  # set motor to 0
        self.ser.ChangeDirection(6)  # set dir to straight

        self.current_speed = self.ser.GetCurrentSpeed()
        self.last_received = self.ser.GetTimeLastReceived()

    def update(self, current_speed, time_received):
        if time_received != self.last_received:
            self.pid.update(current_speed, current_time=time_received)
            self.last_received = time_received
            new_pwm = int(self.pid.output)

            # basic filter
            if new_pwm < self.high_th and new_pwm > -self.low_th:
                self.pwm = new_pwm+self.low_th
            elif new_pwm <= -self.low_th:
                self.pwm -= 5
            else:
                self.pwm = self.high_th

            return self.pwm

        else:
            return None

    def update_ser(self):
        if self.ser is not None:
            return self.update(self.ser.GetCurrentSpeed(),
                               self.ser.GetTimeLastReceived())
        else:
            raise ValueError(
                'Ser is not initialized, please execute init_ser(ser)')

    def changePWM(self, pwm):
        self.ser.changePWM(pwm)

    def update_target(self, new_target):
        self.pid.SetPoint = new_target
