import time

from custom_modules import serial_command2

comPort = '/dev/ttyUSB0'
ser = serial_command2.start_serial(comPort)

if __name__ == '__main__':
    for i in range(50):
        ser.ChangePWM(127+i, 0, 255)
        time.sleep(0.1)

    for i in range(50, 0, -1):
        ser.ChangePWM(127+i, 0, 255)
        time.sleep(0.1)

    for i in range(100):
        ser.ChangePWM(127-i, 0, 255)
        time.sleep(0.1)

    for i in range(100, 0, -1):
        ser.ChangePWM(127-i, 0, 255)
        time.sleep(0.1)
