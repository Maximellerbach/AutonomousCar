import time
from custom_modules import serial_command2

comPort = '/dev/ttyUSB0'
ser = serial_command2.start_serial(comPort)

if __name__ == '__main__':
    for i in range(255):
        ser.ChangeDirection(i, 0, 255)
        time.sleep(0.01)

    for i in range(255, 0, -1):
        ser.ChangeDirection(i, 0, 255)
        time.sleep(0.01)
