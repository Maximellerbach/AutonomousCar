import threading
import time
from enum import IntEnum

import serial

from .sensors import sensor_class

lock = threading.RLock()


class control:
    """This classs send trhu serial port commands to an Arduino to pilot 2 motors using PWM and a servo motor."""

    def __init__(self, port):
        """
        Initialize the class. It does require a serial port name. it can be COMx where x is an interger on Windows.
        Or /dev/ttyXYZ where XYZ is a valid tty output for example /dev/ttyS2 or /dev/ttyUSB0
        """
        self.__ser = serial.Serial()
        self.__sensor_compteTour = sensor_class.CompteTour()
        self.__ser.port = port
        self.__ser.baudrate = 115200
        self.__ser.bytesize = serial.EIGHTBITS  # number of bits per bytes
        self.__ser.parity = serial.PARITY_NONE  # set parity check: no parity
        self.__ser.stopbits = serial.STOPBITS_ONE  # number of stop bits
        self.__ser.timeout = 0 # no timeout
        self.__command = bytearray([0, 0])
        self.__pwm = 0
        self.__steering = 0
        self.__isRuning = True
        self.__isOperation = False
        self.__boosting = False
        self.__toSend = []
        try:
            self.__ser.open()
            print("Serial port open")
            print(self.__ser.portstr)  # check which port was really used
            self.__ser.write(self.__command)
            self.__thread = threading.Thread(target=self.__MainThread__)
            self.__thread.start()
        except Exception as e:
            print("Error opening port: " + str(e))

        time.sleep(1) 

    def __enter__(self):
        return self

    def stop(self):
        self.__isRuning = False
        self.__thread.join()
        if (self.__ser.is_open):
            with lock:
                self.__ser.close()  # close port

    def __safeWrite__(self, command):
        if (self.__ser.is_open):
            while(self.__isOperation):
                pass
            self.__isOperation = True
            self.__ser.write(command)
            self.__ser.flush()
            self.__isOperation = False

    def ChangeSteering(self, steering, min, max):
        """Change steering."""
        self.__command[0] = steering
        self.__toSend.append(self.__command)

    def ChangePWM(self, pwm, min, max):
        """Change motor speed."""
        if (pwm < 0):
            pwm = 0
        if (pwm > 255):
            pwm = 255

        self.__command[1] = pwm
        self.__pwm = pwm
        self.__toSend.append(self.__command)

    def ChangeAll(self, steering, pwm):
        """
        Change all the elements at the same time.

        steering is a byte from 0 to 255.
        PWM is a byte from 0 to 255.
        """
        if (pwm < 0):
            pwm = 0
        if (pwm > 255):
            pwm = 255
        self.__command[0] = steering
        self.__command[1] = pwm
        self.__steering = steering
        self.__pwm = pwm
        self.__toSend.append(self.__command)

    def __MainThread__(self):
        while self.__isRuning:
            for cmd in self.__toSend:
                self.__safeWrite__(cmd)
                self.__toSend.remove(cmd)

    def GetSensor(self):
        return self.__sensor_compteTour

    def GetTurns(self):
        return self.__sensor_compteTour.measurement

    def GetTimeLastReceived(self):
        return self.__sensor_compteTour.time_last_received

    def GetCurrentPosition(self):
        return self.__sensor_compteTour.position

    def GetCurrentSpeed(self):
        return self.__sensor_compteTour.speed

    def GetCurrentAcc(self):
        return self.__sensor_compteTour.acc


def start_serial(port="/dev/ttyUSB0"):
    ser = control(port)
    return ser
