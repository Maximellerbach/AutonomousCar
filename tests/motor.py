import unittest
import time

from custom_modules import serial_command

comPort = '/dev/ttyUSB0'
ser = serial_command.start_serial(comPort)
TEST_PWM = 50


class MotorTestCase(unittest.TestCase):
    def reset(self):
        ser.ChangePWM(0)
        ser.ChangeDirection(6)
        ser.ChangeMotorA(serial_command.Motor.MOTOR_STOP)

    def test_motor_forward(self):
        self.reset()
        ser.ChangeMotorA(serial_command.Motor.MOTOR_FORWARD)
        ser.ChangePWM(TEST_PWM)
        time.sleep(3)
        current_speed = ser.GetCurrentSpeed()
        print(current_speed)
        self.reset()
        time.sleep(1)
        self.assertGreaterEqual(current_speed, 0)

    def test_motor_backward(self):
        self.reset()
        ser.ChangeMotorA(serial_command.Motor.MOTOR_BACKWARD)
        ser.ChangePWM(TEST_PWM)
        time.sleep(3)
        current_speed = ser.GetCurrentSpeed()
        print(current_speed)
        self.reset()
        time.sleep(1)
        self.assertLessEqual(current_speed, 0)



if __name__ == '__main__':
    unittest.main()
    exit()

