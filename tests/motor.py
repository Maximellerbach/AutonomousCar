import unittest
import time

from custom_modules import serial_command

comPort = '/dev/ttyUSB0'
ser = serial_command.start_serial(comPort)
TEST_PWM = 60


class MotorTestCase(unittest.TestCase):
    def reset(self):
        ser.ChangePWM(0)
        ser.ChangeDirection(6)
        ser.ChangeMotorA(serial_command.Motor.MOTOR_STOP)

    def test_motor_forward(self):
        ser.ChangeMotorA(serial_command.Motor.MOTOR_FORWARD)
        ser.ChangePWM(TEST_PWM)
        time.sleep(3)
        self.assertGreaterEqual(ser.GetCurrentSpeed(), 0)
        self.reset()

    def test_motor_backward(self):
        ser.ChangeMotorA(serial_command.Motor.MOTOR_BACKWARD)
        ser.ChangePWM(TEST_PWM)
        time.sleep(3)
        ser.GetCurrentSpeed()
        self.assertLessEqual(ser.GetCurrentSpeed(), 0)
        self.reset()


if __name__ == '__main__':
    unittest.main()
