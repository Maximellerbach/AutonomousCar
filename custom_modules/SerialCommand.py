import serial
from enum import IntEnum
import threading

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
        #ser.writeTimeout = 1     #timeout for write
        self.__command = bytearray([0, 0])
        self.__rounds = 0
        self.__isRuning = True 
        self.__isOperation = False       
        try:
            self.__ser.open()
            print("Serial port open")
            print(self.__ser.portstr)       # check which port was really used
            self.__ser.write(self.__command)
            #self.__ReadTurns__()
            #self.__thread = threading.Thread(target = self.__ReadTurns__)
            self.__thread.start()
        except Exception as e:
            print("Error opening port: " + str(e))

    def __enter__(self):
        return self
    
    def __exit__(self):
        self.__isRuning = False
        self.__thread.join()
        if (self.__ser.is_open):
            with lock:
                self.__ser.close()             # close port

    def __safeWrite__(self, command):
        if (self.__ser.is_open):
            while(self.__isOperation):
                pass
            self.__isOperation = True
            self.__ser.write(command)
            self.__isOperation = False


    def ChangeDirection(self, dir):
        "Change direction, use the direction enum."
        # apply the mask for direction and send the command
        self.__command[0] = (self.__command[0] & 0b11110000) | (dir.to_bytes(1, byteorder='big')[0] & 0b00001111)   
        #print(self.__command[0])
        self.__safeWrite__(self.__command)

    def ChangeMotorA(self, mot):
        "Change motor A state, use the motor enum."
        self.__command[0] = (self.__command[0] & 0b11001111) | ((mot.to_bytes(1, byteorder='big')[0] & 0b000011) << 4)
        #print(self.__command[0])
        self.__safeWrite__(self.__command)

    def ChangeMotorB(self, mot):
        "Change motor A state, use the motor enum."
        self.__command[0] = (self.__command[0] & 0b00111111) | ((mot.to_bytes(1, byteorder='big')[0] & 0b00000011) << 6)
        #print(self.__command[0])
        self.__safeWrite__(self.__command)

    def ChangePWM(self, pwm):
        "Change both motor speed, use byte from 0 to 255."
        if (pwm < 0):
            pwm = 0
        if (pwm > 255):
            pwm = 255
        self.__command[1] = pwm
        self.__safeWrite__(self.__command)

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
        self.__safeWrite__(self.__command)

    def __ReadTurns__(self):    
        while self.__isRuning:
            if self.__ser.in_waiting > 0:
                while(self.__isOperation):
                    pass
                try:
                    self.__isOperation = True
                    out = self.__ser.readlines().split("\r")[-1]
                    if out != '':
                        self.__rounds = out.decode()
                except:
                    pass
                finally:
                    self.__isOperation = False
    
    def GetTurns(self):
        #if self.__ser.inWaiting() > 0:
        #try:
        #    with lock:
        #        if self.__ser.inWaiting() > 0:
        #            out = self.__ser.readlines().split("\r")[-1]
        #            if out != '':
        #                self.__rounds = out.decode()
        #except Exception as e:
        #    print("Error opening port: " + str(e))

        return self.__rounds

