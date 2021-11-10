import threading
import time

import serial

lock = threading.RLock()


def map_value(value, min, max, outmin, outmax):
    if value < min:
        value = min
    elif value > max:
        value = max

    return ((outmax - outmin) * (value - min)) / (max - min) + outmin


class control:
    """This classs send trhu serial port commands to an Arduino to pilot 2 motors using PWM and a servo motor."""

    def __init__(self, port):
        """
        Initialize the class. It does require a serial port name. it can be COMx where x is an interger on Windows.
        Or /dev/ttyXYZ where XYZ is a valid tty output for example /dev/ttyS2 or /dev/ttyUSB0
        """
        self.__ser = serial.Serial()
        self.__ser.port = port
        self.__ser.baudrate = 115200
        self.__ser.bytesize = serial.EIGHTBITS  # number of bits per bytes
        self.__ser.parity = serial.PARITY_NONE  # set parity check: no parity
        self.__ser.stopbits = serial.STOPBITS_ONE  # number of stop bits
        self.__ser.timeout = 0  # no timeout

        self.__sensor_rpm = 0  # init rpm of the sensor to 0
        self.__command = bytearray([255, 127, 127, 0])
        self.__isRuning = True
        self.__isOperation = False
        self.__boosting = False
        self.__toSend = []
        try:
            self.__ser.open()
            print("Serial port open")
            print(self.__ser.portstr)  # check which port was really used
            self.__ser.write(self.__command)
        except Exception as e:
            print("Error opening port: " + str(e))

        self.__thread = threading.Thread(target=self.__runThreaded__)
        self.__thread.start()

        time.sleep(1)

    def __runThreaded__(self):
        while(self.__isRuning and self.__ser.is_open):
            for cmd in self.__toSend:
                self.__safeWrite__(cmd)
                self.__toSend.remove(cmd)
            self.__readRPM__()

    def stop(self):
        self.__isRuning = False
        self.__thread.join()
        if self.__ser.is_open:
            with lock:
                self.__ser.close()  # close port

    def __safeWrite__(self, command):
        if self.__ser.is_open:
            while self.__isOperation:
                pass
            self.__isOperation = True
            self.__ser.write(command)
            self.__ser.flush()
            self.__isOperation = False

    def __readRPM__(self):
        print(self.__ser.in_waiting)

        # while self.__isOperation:
        #     pass
        # self.__isOperation = True
        # try:
        #     out = self.__ser.readlines()[-1]
        #     print(out)
        # except:
        #     pass
        # finally:
        #     self.__isOperation = False


        # if self.__ser.in_waiting > 0:
        #     out = self.__ser.readlines()[-1]
        #     if out != "" or out is not None:
        #         print(out)
        #         # __sensor_rpm = int(out)
        #     else:
        #         print("out is empty")

    def ChangeDirection(self, steering, min=-1, max=1):
        """Change steering."""
        steering = int(map_value(steering, min, max, 0, 255))
        self.__command[1] = steering
        # self.__ser.write(self.__command)
        self.__toSend.append(self.__command)

    def ChangePWM(self, pwm, min=-1, max=1):
        """Change motor speed."""
        pwm = int(map_value(pwm, min, max, 0, 255))
        self.__command[2] = pwm
        # self.__ser.write(self.__command)
        self.__toSend.append(self.__command)

    def ChangeAll(self, steering, pwm, min=[-1, -1], max=[1, 1]):
        """
        Change all the elements at the same time.

        steering is a byte from 0 to 255.
        PWM is a byte from 0 to 255.
        """
        steering = int(map_value(steering, min[0], max[0], 0, 255))
        pwm = int(map_value(pwm, min[1], max[1], 0, 255))

        self.__command[1] = steering
        self.__command[2] = pwm
        # self.__ser.write(self.__command)
        self.__toSend.append(self.__command)

    def GetRPM(self):
        return self.__sensor_rpm


def start_serial(port="/dev/ttyUSB0"):
    ser = control(port)
    return ser


if __name__ == "__main__":
    # motor test, servo test and rpm test
    ser = start_serial()
    # ChangeAll()
