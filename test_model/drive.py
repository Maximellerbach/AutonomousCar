import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.eviron['openmp'] = 'True'
import time

import cv2
from custom_modules import architectures, serial_command2
from custom_modules.datasets import dataset_json

core_count = 4
architectures.tf.config.threading.set_inter_op_parallelism_threads(core_count)
architectures.tf.config.threading.set_intra_op_parallelism_threads(core_count)

serialport = "/dev/ttyUSB0"
os.system("sudo chmod 0666 {}".format(serialport))
ser = serial_command2.control(serialport)

MAXTHROTTLE = 1
wi = 160
he = 120

Dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
input_components = []

cap = cv2.VideoCapture(0)
ret, img = cap.read()  # read the camera once to make sure it works
assert ret is True

basedir = os.path.dirname(os.path.abspath(__file__))
model = architectures.safe_load_model(f"{basedir}/models/auto_label7.h5", compile=False)
architectures.apply_predict_decorator(model)

print("Starting mainloop")

while True:
    try:
        st = time.time()

        _, cam = cap.read()
        img = cv2.resize(cam, (wi, he))

        memory = {}
        memory["direction"] = 0
        memory["speed"] = 0
        memory["throttle"] = 0.1
        memory["time"] = time.time()

        to_pred = Dataset.make_to_pred_annotations([img], [memory], input_components)

        # PREDICT
        prediction_dict, elapsed_time = model.predict(to_pred)
        prediction_dict = prediction_dict[0]
        memory["direction"] = prediction_dict["direction"]

        ser.ChangeAll(memory["direction"], MAXTHROTTLE * memory["throttle"])

        dt = time.time() - st
        print(prediction_dict, 1 / elapsed_time, 1 / dt)

    except Exception as e:
        print(e)

    except KeyboardInterrupt:
        break

ser.ChangeAll(0, 0)
cap.release()
