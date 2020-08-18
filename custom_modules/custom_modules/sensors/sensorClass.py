import time
import numpy as np
from ..SerialCommand import car

def transform_axes(axes, multiplier):
    return axes*multiplier


def derivate_axes(axes, dt, axes_len=None):
    axes_len = axes_len if axes_len is not None else axes.shape
    return axes/np.full(axes_len, dt)


def integrate_axes(integration, axes, dt, axes_len=None):
    axes_len = axes_len if axes_len is not None else axes.shape
    return integration+axes*np.full(axes_len, dt)


class sensor_compteTour():
    AXES_TRANSFORMER = np.array([(1/84)*car.REAR_PERIMETER])
    MEA_ERROR = np.array([0.05])
    INITIAL_STATE = np.array([0])
    DATA_LEVEL = 0  # (metric, speed, acc)

    measurement = position = speed = acc = INITIAL_STATE

    datas = (position, speed, acc)
    time_last_received = time.time()

    @classmethod
    def update(cls, new_measurement, new_time=None):
        new_time = new_time if new_time is not None else time.time()
        new_measurement = transform_axes(
            np.array(new_measurement), cls.AXES_TRANSFORMER)

        dt = new_time-cls.time_last_received

        new_speed = derivate_axes(new_measurement-cls.datas[0], dt)
        new_acc = derivate_axes(new_speed-cls.datas[1], dt)

        cls.position = cls.position+new_measurement
        cls.speed = new_speed
        cls.acc = new_acc

        cls.datas = (cls.position, cls.speed, cls.acc)
        cls.time_last_received = new_time
        cls.measurement = new_measurement


class sensor_accelerometer():
    AXES_TRANSFORMER = np.array([1, 1, 1])
    MEA_ERROR = np.array([0.05, 0.05, 0.05])
    INITIAL_STATE = np.array([0, 0, 0])
    DATA_LEVEL = 2  # (metric, speed, acc)

    measurement = position = speed = acc = INITIAL_STATE

    datas = (position, speed, acc)
    time_last_received = time.time()

    @classmethod
    def update(cls, new_measurement, new_time=None):
        new_time = new_time if new_time is not None else time.time()
        new_measurement = transform_axes(
            np.array(new_measurement), cls.AXES_TRANSFORMER)

        dt = new_time-cls.time_last_received

        new_speed = integrate_axes(cls.speed, new_measurement, dt)
        new_position = integrate_axes(cls.position, new_speed, dt)

        cls.position = new_position
        cls.speed = new_speed
        cls.acc = new_measurement

        cls.datas = (cls.position, cls.speed, cls.acc)
        cls.time_last_received = new_time
        cls.measurement = new_measurement


class sensor_magnetometer():
    AXES_TRANSFORMER = np.array([1, 1, 1])
    MEA_ERROR = np.array([0.05, 0.05, 0.05])
    INITIAL_STATE = np.array([0, 0, 0])
    DATA_LEVEL = 0  # (metric, speed, acc)

    measurement = position = speed = acc = INITIAL_STATE

    datas = (position, speed, acc)
    time_last_received = time.time()

    @classmethod
    def update(cls, new_measurement, new_time=None):
        new_time = new_time if new_time is not None else time.time()
        new_measurement = transform_axes(
            np.array(new_measurement), cls.AXES_TRANSFORMER)

        dt = new_time-cls.time_last_received

        new_speed = derivate_axes(new_measurement-cls.datas[0], dt)
        new_acc = derivate_axes(new_speed-cls.datas[1], dt)

        cls.position = cls.position+new_measurement
        cls.speed = new_speed
        cls.acc = new_acc

        cls.datas = (cls.position, cls.speed, cls.acc)
        cls.time_last_received = new_time
        cls.measurement = new_measurement


class sensor_fusion():
    sensors = []

    @classmethod
    def get_sensor_by_index(cls, index):
        return cls.sensors[index]


if __name__ == "__main__":
    # some tests
    sensor = sensor_compteTour()
    time.sleep(0.1)
    sensor.update([100])
    print(sensor.datas)
    time.sleep(0.2)
    sensor.update([200])
    print(sensor.datas)
