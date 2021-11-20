import os
import time

import cv2
from custom_modules import architectures, serial_command2, camera, memory
from custom_modules.datasets import dataset_json

import controller


def deadzone(value, th, default=0):
    return value if abs(value) > th else default


dos_save = os.path.expanduser("~") + "/recorded/"
if not os.path.isdir(dos_save):
    os.mkdir(dos_save)

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = [1]

Memory = memory.Memory(Dataset, dos_save, queue_size=10)

serialport = "/dev/ttyUSB0"
os.system("sudo chmod 0666 {}".format(serialport))
ser = serial_command2.control(serialport)

MAXTHROTTLE = 0.25
th_steering = 0.05  # 5% threshold
th_throttle = 0.06  # 6% threshold
wi = 160
he = 120

joy = controller.XboxOneJoystick()
joy.init()
assert joy.connected is True
print("joy working")


# cap = camera.usbWebcam(topcrop=0.2, botcrop=0.0)
cap = cv2.VideoCapture(0)
print("cam working")

basedir = os.path.dirname(os.path.abspath(__file__))

# model = architectures.safe_load_model(f"{basedir}/models/auto_label7.h5", compile=False)

# Load TFLite model
model = architectures.safe_load_model(f"{basedir}/../test_model/models/working_renault2.tflite", ["direction", "throttle"])

# checking if the controller is working properly
joy_leftX = 0
while joy_leftX <= 0.9:
    joy_leftX = joy.axis_states["x"]
    print(joy_leftX, end="\r")
    time.sleep(0.01)

while joy_leftX >= -0.9:
    joy_leftX = joy.axis_states["x"]
    print(joy_leftX, end="\r")
    time.sleep(0.01)
    
print("Starting mainloop")

Memory.run()
ret = True

while not joy.button_states["back"] and joy.connected and ret:
    joy_steering = joy.axis_states["x"]
    joy_throttle = joy.axis_states["rz"]
    joy_brake = joy.axis_states["z"]
    joy_button_a = joy.button_states["a"]
    joy_button_x = joy.button_states["x"]

    st = time.time()
    _, cam = cap.read()
    img = cv2.resize(cam, (wi, he))

    annotation = {}
    annotation["direction"] = 0
    annotation["speed"] = ser.GetSpeed()
    annotation["throttle"] = 0.2
    annotation["time"] = st

    if joy_button_x or joy_button_a:  # Manual steering
        annotation["direction"] = deadzone(joy_steering, th_steering)
        annotation["throttle"] = deadzone(joy_throttle - joy_brake, th_throttle)
        if not joy_button_x:  # Record
            Memory.add(img, annotation)

    else:  # Do the prediction
        to_pred = Dataset.make_to_pred_annotations([img], [annotation], input_components)

        prediction_dict, elapsed_time = model.predict(to_pred)
        annotation["direction"] = prediction_dict["direction"]
        annotation["throttle"] = prediction_dict["throttle"]

        dt = time.time() - st
        print(prediction_dict, 1 / elapsed_time, 1 / dt)

    # apply direction and throttle
    ser.ChangeAll(annotation["direction"], MAXTHROTTLE * annotation["throttle"])


Memory.stop()
ser.ChangeAll(0, 0)
cap.release()

if not joy.connected:
    print("Lost connection with joystick")
elif not ret:
    print("Camera isn't working properly")
else:
    print("Terminated")

joy.connected = False
