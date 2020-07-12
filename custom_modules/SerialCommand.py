import math
import threading
import time
from enum import IntEnum

import serial

lock = threading.RLock()

class direction(IntEnum):
    DIR_LEFT_7 = 0
    DIR_LEFT_6 = 1
    DIR_LEFT_5 = 2
    DIR_LEFT_4 = 3
    DIR_LEFT_3 = 4
    DIR_LEFT_2 = 5
    DIR_LEFT_1 = 6
    DIR_STRAIGHT = 7
    DIR_RIGHT_1 = 8
    DIR_RIGHT_2 = 9
    DIR_RIGHT_3 = 10
    DIR_RIGHT_4 = 11
    DIR_RIGHT_5 = 12
    DIR_RIGHT_6 = 13
    DIR_RIGHT_7 = 14

class motor(IntEnum):
    MOTOR_STOP = 0
    MOTOR_FORWARD = 1
    MOTOR_BACKWARD = 2
    MOTOR_IDLE = 3

class car():
    WHEEL_BASE = 0.257
    REAR_DIAMETER = 0.105
    FRONT_DIAMETER = 0.082
    REAR_PERIMETER = REAR_DIAMETER*math.pi
    FRONT_PERIMETER = FRONT_DIAMETER*math.pi
    SENSOR_RATIO = 1/(14*6)

class control:    
    "This classs send trhu serial port commands to an Arduino to pilot 2 motors using PWM and a servo motor"
    def __init__(self, port):
        "Initialize the class. It does require a serial port name. it can be COMx where x is an interger on Windows. Or /dev/ttyXYZ where XYZ is a valid tty output for example /dev/ttyS2 or /dev/ttyUSB0"
        self.__ser = serial.Serial()
        self.__ser.port = port
        self.__ser.baudrate = 115200
        self.__ser.bytesize = serial.EIGHTBITS #number of bits per bytes
        self.__ser.parity = serial.PARITY_NONE #set parity check: no parity
        self.__ser.stopbits = serial.STOPBITS_ONE #number of stop bits
        self.__ser.timeout = 0     #no timeout
        self.__command = bytearray([0, 0])
        self.__rounds = 0
        self.__current_speed = 0
        self.__time_last_received = time.time()
        self.__isRuning = True 
        self.__isOperation = False
        self.__toSend = []
        try:
            self.__ser.open()
            print("Serial port open")
            print(self.__ser.portstr)       # check which port was really used
            self.__ser.write(self.__command)
            self.__thread = threading.Thread(target = self.__ReadTurns__)
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
                self.__ser.close() # close port

    def __safeWrite__(self, command):
        if (self.__ser.is_open):
            while(self.__isOperation):
                pass
            self.__isOperation = True
            self.__ser.write(command)
            self.__ser.flush()
            self.__isOperation = False


    def ChangeDirection(self, dir):
        "Change direction, use the direction enum."
        # apply the mask for direction and send the command
        self.__command[0] = (self.__command[0] & 0b11110000) | (dir.to_bytes(1, byteorder='big')[0] & 0b00001111)
        self.__toSend.append(self.__command)

    def ChangeMotorA(self, mot):
        "Change motor A state, use the motor enum."
        self.__command[0] = (self.__command[0] & 0b11001111) | ((mot.to_bytes(1, byteorder='big')[0] & 0b000011) << 4)
        self.__toSend.append(self.__command)

    def ChangeMotorB(self, mot):
        "Change motor A state, use the motor enum."
        self.__command[0] = (self.__command[0] & 0b00111111) | ((mot.to_bytes(1, byteorder='big')[0] & 0b00000011) << 6)
        self.__toSend.append(self.__command)

    def ChangePWM(self, pwm):
        "Change both motor speed, use byte from 0 to 255."
        if (pwm < 0):
            pwm = 0
        if (pwm > 255):
            pwm = 255
        self.__command[1] = pwm
        self.__toSend.append(self.__command)

    def ChangeAll(self, dir, motorA, motorB, pwm):
        "Change all the elements at the same time. Consider using the direction and motor enums. PWM is byte from 0 to 255."
        self.__command[0] = (self.__command[0] & 0b11110000) | (dir.to_bytes(1, byteorder='big')[0] & 0b00001111)
        self.__command[0] = (self.__command[0] & 0b11001111) | ((motorA.to_bytes(1, byteorder='big')[0] & 0b00000011) << 4)
        self.__command[0] = (self.__command[0] & 0b00111111) | ((motorB.to_bytes(1, byteorder='big')[0] & 0b00000011) << 6)
        if (pwm < 0):
            pwm = 0
        if (pwm > 255):
            pwm = 255
        self.__command[1] = pwm
        self.__toSend.append(self.__command)

    def __ReadTurns__(self):
        while self.__isRuning:
            for cmd in self.__toSend:
                self.__safeWrite__(cmd)
                self.__toSend.remove(cmd)
            if self.__ser.in_waiting > 0:
                while(self.__isOperation):
                    pass
                self.__isOperation = True
                try:
                    out = self.__ser.readlines()[-1]
                    if out != '':
                        new_rounds = -int(out.decode())
                        new_time = time.time()
                        dt = new_time-self.__time_last_received
                        dturn = new_rounds-self.__rounds

                        if abs(dturn) > 16384:
                            # dturn = new_rounds-(self.__rounds-65536)
                            self.__rounds = new_rounds
                            print("OVERFLOW DETECTED")
                        else:
                            self.__current_speed = (car.REAR_PERIMETER*(dturn*car.SENSOR_RATIO))/dt
                            self.__time_last_received = new_time
                            self.__rounds = new_rounds
                except:
                    pass
                finally:
                    self.__isOperation = False

    def GetTurns(self):
        return self.__rounds

    def GetTimeLastReceived(self):
        return self.__time_last_received

    def GetCurrentSpeed(self):
        return self.__current_speed