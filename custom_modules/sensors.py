import math
import time
import numpy as np


class car():
    WHEEL_BASE = 0.257
    REAR_DIAMETER = 0.105
    FRONT_DIAMETER = 0.082
    REAR_PERIMETER = REAR_DIAMETER*math.pi
    FRONT_PERIMETER = FRONT_DIAMETER*math.pi


def transform_axes(axes, multiplier):
    return axes*multiplier


def derivate_axes(axes, dt, axes_len=None):
    axes_len = axes_len if axes_len is not None else axes.shape
    return axes/np.full(axes_len, dt)


def integrate_axes(integration, axes, dt, axes_len=None):
    axes_len = axes_len if axes_len is not None else axes.shape
    return integration+axes*np.full(axes_len, dt)


class sensor_compteTour():
    def __init__(self):
        self.AXES_TRANSFORMER = np.array([(1/84)*car.REAR_PERIMETER])
        self.MEA_ERROR = np.array([0.05])
        self.INITIAL_STATE = np.array([0])
        self.DATA_LEVEL = 0  # (metric, speed, acc)

        self.measurement = self.INITIAL_STATE
        self.position = self.INITIAL_STATE
        self.speed = self.INITIAL_STATE
        self.acc = self.INITIAL_STATE

        self.datas = (self.position, self.speed, self.acc)
        self.time_last_received = time.time()

    def update(self, new_measurement, new_time=None):
        new_time = new_time if new_time is not None else time.time()
        new_measurement = transform_axes(np.array(new_measurement), self.AXES_TRANSFORMER)

        dt = new_time-self.time_last_received

        new_speed = derivate_axes(new_measurement-self.datas[0], dt)
        new_acc = derivate_axes(new_speed-self.datas[1], dt)

        self.position = self.position+new_measurement
        self.speed = new_speed
        self.acc = new_acc

        self.datas = (self.position, self.speed, self.acc)
        self.time_last_received = new_time
        self.measurement = new_measurement


class sensor_accelerometer():
    def __init__(self):
        self.AXES_TRANSFORMER = np.array([1, 1, 1])
        self.MEA_ERROR = np.array([0.05, 0.05, 0.05])
        self.INITIAL_STATE = np.array([0, 0, 0])
        self.DATA_LEVEL = 2  # (metric, speed, acc)

        self.measurement = self.INITIAL_STATE
        self.position = self.INITIAL_STATE
        self.speed = self.INITIAL_STATE
        self.acc = self.INITIAL_STATE

        self.datas = (self.position, self.speed, self.acc)
        self.time_last_received = time.time()

    def update(self, new_measurement, new_time=None):
        new_time = new_time if new_time is not None else time.time()
        new_measurement = transform_axes(np.array(new_measurement), self.AXES_TRANSFORMER)

        dt = new_time-self.time_last_received

        new_speed = integrate_axes(self.speed, new_measurement, dt)
        new_position = integrate_axes(self.position, new_speed, dt)

        self.position = new_position
        self.speed = new_speed
        self.acc = new_measurement

        self.datas = (self.position, self.speed, self.acc)
        self.time_last_received = new_time
        self.measurement = new_measurement


class sensor_magnetometer():
    def __init__(self):
        self.AXES_TRANSFORMER = np.array([1, 1, 1])
        self.MEA_ERROR = np.array([0.05, 0.05, 0.05])
        self.INITIAL_STATE = np.array([0, 0, 0])
        self.DATA_LEVEL = 0 # (metric, speed, acc)

        self.measurement = self.INITIAL_STATE
        self.position = self.INITIAL_STATE
        self.speed = self.INITIAL_STATE
        self.acc = self.INITIAL_STATE

        self.datas = (self.position, self.speed, self.acc)
        self.time_last_received = time.time()

    def update(self, new_measurement, new_time=None):
        new_time = new_time if new_time is not None else time.time()
        new_measurement = transform_axes(np.array(new_measurement), self.AXES_TRANSFORMER)

        dt = new_time-self.time_last_received

        new_speed = derivate_axes(new_measurement-self.datas[0], dt)
        new_acc = derivate_axes(new_speed-self.datas[1], dt)

        self.position = self.position+new_measurement
        self.speed = new_speed
        self.acc = new_acc

        self.datas = (self.position, self.speed, self.acc)
        self.time_last_received = new_time
        self.measurement = new_measurement


class sensor_fusion():
    def __init__(self, sensors):
        self.sensors = sensors

    def get_sensor_by_index(self, index):
        return self.sensors[index]


if __name__ == "__main__":
    # some tests
    sensor = sensor_compteTour()
    time.sleep(0.1)
    sensor.update([100])
    print(sensor.datas)
    time.sleep(0.2)
    sensor.update([200])
    print(sensor.datas)
