import os
import time

from custom_modules import serial_command2
from custom_modules.datasets import dataset_json

import cv2
import controller


def deadzone(value, th, default=0):
    return value if abs(value) > th else default


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])

dos_save = os.path.expanduser('~') + "/recorded/"
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXTHROTTLE = 0.5
th_steering = 0.05  # 5% threshold
th_throttle = 0.06  # 6% threshold

comPort = "/dev/ttyUSB0"
ser = serial_command2.start_serial(comPort)
joy = controller.XboxOneJoystick()
joy.init()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

ret, img = cap.read()  # read the camera once to make sure it works
assert ret is True
print(img.shape)
assert img.shape == (120, 160, 3)


# checking if the controller is working properly
joy_leftX = 0
while joy_leftX <= 0.9:
    joy_leftX = joy.axis_states['x']
    print(joy_leftX, end="\r")
    time.sleep(0.01)

while joy_leftX >= -0.9:
    joy_leftX = joy.axis_states['x']
    print(joy_leftX, end="\r")
    time.sleep(0.01)

print("Starting mainloop")

while not joy.button_states['back'] and joy.connected:
    joy_steering = joy.axis_states['x']
    joy_throttle = joy.axis_states['rz']
    joy_brake = joy.axis_states['z']
    joy_button_a = joy.button_states['a']

    memory = {}

    memory['direction'] = deadzone(joy_steering, th_steering)
    memory['throttle'] = deadzone(joy_throttle - joy_brake, th_throttle)
    memory['speed'] = 0
    memory['time'] = time.time()

    ser.ChangeAll(memory['direction'],
                  MAXTHROTTLE * memory['throttle'],
                  min=[-1, -1], max=[1, 1])

    if joy_button_a:  # save the image
        _, img = cap.read()

        Dataset.save_img_and_annotation(
            img,
            annotation=memory(),
            dos=dos_save
        )


# stop steering and throttle
ser.ChangeAll(0, 0)

if not joy.connected:
    print("Lost connection with joystick")
else:
    print('Terminated')
