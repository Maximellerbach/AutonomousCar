import os
import time

import cv2
from custom_modules import architectures, camera, serial_command2
from custom_modules.datasets import dataset_json


def get_key_by_name(dict, name):
    for k in dict.keys():
        if name in k:
            return dict[k]
    return None


serialport = "/dev/ttyUSB0"
os.system("sudo chmod 0666 {}".format(serialport))
ser = serial_command2.control(serialport)

MAXTHROTTLE = 1
wi = 160
he = 120

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = []

cap = camera.usbWebcam()

basedir = os.path.dirname(os.path.abspath(__file__))
# model = architectures.safe_load_model(
#     f"{basedir}/models/auto_label7.h5", compile=False)
# architectures.apply_predict_decorator(model)

model = architectures.TFLite(f"{basedir}/models/auto_label7.tflite", ["direction"])

cap.start()
print("Starting mainloop")

while True:
    try:
        st = time.time()

        cam = cap.read()
        img = cv2.resize(cam, (wi, he))

        memory = {}
        memory["direction"] = 0
        memory["speed"] = 0
        memory["throttle"] = 0.1
        memory["time"] = time.time()

        to_pred = Dataset.make_to_pred_annotations(
            [img], [memory], input_components)

        # PREDICT
        prediction_dict, elapsed_time = model.predict(to_pred)

        if isinstance(prediction_dict, list):
            prediction_dict = prediction_dict[0]
        memory["direction"] = get_key_by_name(prediction_dict, "direction")

        ser.ChangeAll(memory["direction"], MAXTHROTTLE * memory["throttle"])

        dt = time.time() - st
        print(prediction_dict, 1 / elapsed_time, 1 / dt)

    except Exception as e:
        print(e)

    except KeyboardInterrupt:
        break

ser.ChangeAll(0, 0)
cap.release()
