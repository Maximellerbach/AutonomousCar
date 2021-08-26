import os
import time

from custom_modules import serial_command2, memory
from custom_modules.datasets import dataset_json

import cv2
import xbox_controller


def deadzone(value, th, default=0):
    return value if abs(value) > th else default


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
memory = memory.Memory(2)

dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXTHROTTLE = 0.5
th_steering = 0.05  # 5% threshold
th_throttle = 0.06  # 6% threshold

comPort = "/dev/ttyUSB0"
ser = serial_command2.start_serial(comPort)
joy = xbox_controller.XboxController()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)


# checking if the controller is working properly
joy_leftX = 0
while joy_leftX != 1.0:
    joy_leftX = joy.LeftJoystickX
    print(joy_leftX)

while joy_leftX != -1.0:
    joy_leftX = joy.LeftJoystickX
    print(joy_leftX)

print("Starting mainloop")

prev_throttle = 0
while not joy.Back():
    joy_steering = joy.LeftJoystickX
    joy_throttle = joy.RightTrigger
    joy_brake = joy.LeftTrigger
    joy_button_a = joy.A

    memory['steering'] = deadzone(joy_steering, th_steering)
    memory['throttle'] = deadzone(joy_throttle - joy_brake, th_throttle)
    memory['speed'] = 0
    memory['time'] = time.time()

    ser.ChangeAll(memory['steering'],
                  MAXTHROTTLE * memory['throttle'],
                  min=[-1, -1], max=[1, 1])

    if joy_button_a:
        _, img = cap.read()

        Dataset.save_img_and_annotation(
            img,
            annotation=memory(),
            dos=dos_save
        )

    memory.append({})

print('terminated')
