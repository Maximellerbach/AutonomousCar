import time
import numpy as np
import sys
from threading import Thread
from di_sensors.inertial_measurement_unit import InertialMeasurementUnit
import math


class sensor():

    def __init__(self, lookback, calib_it):

        self.imu = InertialMeasurementUnit()
        stat = self.imu.BNO055.get_calibration_status()
        # print(stat)

        calib = self.imu.BNO055.get_calibration()
        # print(calib)

        calib = []
        gyro_calib = []

        sat = time.time()
        for _ in range(calib_it):
            calib.append(self.imu.read_linear_acceleration())
            gyro_calib.append(self.imu.read_gyroscope())

        eat = time.time()

        self.dat = (eat-sat)/calib_it
        self.calib = np.average(calib, axis=0)
        print(self.calib)

        self.gyro_calib = np.average(gyro_calib, axis=0)
        print(self.gyro_calib)

        self.lookback = lookback
        self.v = 0.0

    def loop(self):

        acc = []
        gyro = []
        lookback = self.lookback
        c = 0
        st = time.time()

        while(True):
            try:

                et = time.time()
                dt = et-st

                st = time.time()

                ax, ay, az = self.imu.read_linear_acceleration()
                gx, gy, gz = self.imu.read_gyroscope()

                if c > lookback:
                    acc[:-1] = acc[1:]
                    acc[-1] = [ax, ay, az]
                    gyro[:-1] = gyro[1:]
                    gyro[-1] = [gx, gy, gz]

                    avacc = np.average(acc, axis=0)
                    avgy = np.average(gyro, axis=0)

                    vect = math.sqrt(avacc[0]**2+avacc[1]**2)

                    if avacc[0] > 0:
                        self.v = 0.99*(self.v + vect * dt)
                    else:
                        self.v = 0.99*(self.v - vect * dt)

                    # if np.abs(avgy[2]-self.gyro_calib[2])>0.1:
                    # 	self.v += 0.01 * avacc[1] / (avgy[2]-self.gyro_calib[2])

                    # x = 1/2* avacc[0] * (dt*dt) + self.pos[0] + self.v[0]
                    # y = 1/2* avacc[1] * (dt*dt) + self.pos[1] + self.v[1]
                    # z = 1/2* avacc[2] * (dt*dt) + self.pos[2] + self.v[2]
                    # self.pos = [x, y, z]

                    # print(pos)

                else:
                    acc.append([ax, ay, az])
                    gyro.append([gx, gy, gz])

                c += 1

            except Exception as e:
                print(e)

    def get_vel(self):
        return self.v


if __name__ == "__main__":

    vel_sensor = sensor(10, 1000)

    thread = Thread(target=vel_sensor.loop)
    thread.start()

    while(True):
        time.sleep(0.1)
        print(vel_sensor.get_vel())
