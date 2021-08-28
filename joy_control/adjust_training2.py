import os
import time

import cv2
from custom_modules import architectures, serial_command2
from custom_modules.datasets import dataset_json

import controller


def deadzone(value, th, default=0):
    return value if abs(value) > th else default


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = []

dos_save = os.path.expanduser('~') + "/recorded/"
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXTHROTTLE = 0.5
th_steering = 0.05  # 5% threshold
th_throttle = 0.06  # 6% threshold

serialport = "/dev/ttyUSB0"
os.system('sudo chmod 0666 {}'.format(serialport))
ser = serial_command2.start_serial(serialport)

joy = controller.XboxOneJoystick()
joy.init()
assert joy.connected is True
print("joy working")

cap = cv2.VideoCapture(0)
ret, img = cap.read()  # read the camera once to make sure it works
assert ret is True
print("cam working")

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.TFLite(
    os.path.normpath(f'{basedir}/auto_label6.tflite'), ['direction'])

print("Starting mainloop")

while not joy.button_states['back'] and joy.connected:

    try:
        joy_steering = joy.axis_states['x']
        joy_throttle = joy.axis_states['rz']
        joy_brake = joy.axis_states['z']
        joy_button_a = joy.button_states['a']
        joy_button_x = joy.button_states['x']

        _, img = cap.read()
        img = cv2.resize(img, (160, 120))

        # annotation template with just what is needed for the prediction
        annotation = {
            'direction': deadzone(joy_steering, th_steering),
            'speed': 0,
            'throttle': deadzone(joy_throttle - joy_brake, th_throttle),
            'time': time.time()
        }

        if joy_button_a or joy_button_x:

            if not joy_button_x:
                Dataset.save_img_and_annotation(
                    img,
                    annotation=annotation,
                    dos=dos_save)

            ser.ChangeAll(annotation['direction'], annotation['throttle'])

        else:
            to_pred = Dataset.make_to_pred_annotations(
                [img], [annotation], input_components)

            predicted, dt = model.predict(to_pred)
            print(predicted)
            ser.ChangeAll(predicted['direction'], MAXTHROTTLE * joy_throttle)

    except Exception as e:
        print(e)

    except KeyboardInterrupt:
        break

ser.ChangeAll(0, 0)

if not joy.connected:
    print("Lost connection with joystick")
else:
    print('Terminated')
