import cv2
import os
import time

import xbox
import utils
from custom_modules import serial_command, architectures
from custom_modules.datasets import dataset_json


Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = [1]

dos_save = os.getcwd()+os.path.normpath("/recorded/")
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

MAXSPEED = 120
th_direction = 0.05
th_throttle = 0.1

comPort = utils.get_port_name()
ser = serial_command.start_serial(comPort)
joy = xbox.Joystick()
cap = cv2.VideoCapture(0)

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.safe_load_model(
    os.path.normpath(f'{basedir}/models/rbrl_sim4_working.h5'))
architectures.apply_predict_decorator(model)

ser.ChangeMotorA(serial_command.Motor.MOTOR_FORWARD)
while not joy.Back():

    joy_steering, joy_throttle, joy_brake, joy_button_a, joy_button_x = utils.get_controller_buttons(
        joy)
    current_speed = ser.GetCurrentSpeed()

    _, img = cap.read()
    # annotation template with just what is needed for the prediction
    annotation = {
        'direction': 0,
        'speed': current_speed,
        'throttle': 0,
        'time': time.time()
    }

    if joy_button_a:

        steering = joy_steering if abs(joy_steering) > abs(
            th_direction) and not joy_button_x else 0
        throttle = joy_throttle - \
            joy_brake if abs(joy_throttle) > abs(th_throttle) else 0

        annotation = {
            'direction': steering,
            'speed': current_speed,
            'throttle': throttle,
            'time': time.time()
        }
        Dataset.save_img_and_annotation(img, annotation)
    else:
        to_pred = Dataset.make_to_pred_annotations(
            [img], [annotation], input_components)

        output_dict, elapsed_time = model.predict(to_pred)
        output_dict = output_dict[0]
        for key in output_dict:
            annotation[key] = output_dict[key]

    # TODO: refactor this to use the speed captured from the wheel
    pwm = int(MAXSPEED * annotation['throttle'])
    ser.ChangeDirection(utils.direction2categorical(steering))
    ser.ChangePWM(pwm)


joy.close()
print('terminated')
