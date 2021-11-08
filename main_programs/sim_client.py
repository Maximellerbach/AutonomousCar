import base64
import json
import os
import threading
import time
from collections.abc import Iterable
from io import BytesIO

import cv2
import keyboard
import numpy as np
import tensorflow
from custom_modules import architectures
from custom_modules.datasets import dataset_json
from custom_modules.visual_interface import AutoInterface, windowInterface
from gym_donkeycar.core.sim_client import SDClient
from PIL import Image

physical_devices = tensorflow.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


def transform_st(st, sq, mult):
    if st >= 0:
        return (st * mult) ** sq
    return -np.absolute(st * mult) ** sq


def opt_acc(st, current_speed, max_throttle, min_throttle, target_speed):
    dt_throttle = max_throttle - min_throttle

    optimal_acc = (target_speed - current_speed) / target_speed
    if optimal_acc < 0:
        optimal_acc = 0

    optimal_acc = min_throttle + ((optimal_acc ** 0.1) * (1 - np.absolute(st))) * dt_throttle

    if optimal_acc > max_throttle:
        optimal_acc = max_throttle
    elif optimal_acc < min_throttle:
        optimal_acc = min_throttle

    return optimal_acc


def cat2linear(cat, coef=[-1, -0.5, 0, 0.5, 1], av=False):

    if av:
        st = 0
        count = 0
        for k in cat[0]:
            if k:
                count += 1

        if count != 0:
            for ait, nyx in enumerate(cat[0]):
                st += nyx * coef[ait]
            st /= count

    else:
        st = 0
        for ait, nyx in enumerate(cat[0]):
            st += nyx * coef[ait]
    return st


class SimpleClient(SDClient):
    def __init__(
        self,
        address,
        model,
        dataset,
        input_components,
        name="0",
        PID_settings=(15, 0.5, 0.0, 1.0, 1.0),
        buffer_time=0.1,
        sleep_time=0.01,
    ):

        super().__init__(*address, poll_socket_sleep_time=sleep_time)
        self.model = model
        self.dataset = dataset
        self.input_components = input_components

        self.name = str(name)
        self.PID_settings = PID_settings
        self.buffer_time = buffer_time

        try:
            self.input_names = dataset.indexes2components_names(input_components)
            self.output_names = architectures.get_model_output_names(model)
        except:
            pass

        self.default_dos = f"C:\\Users\\maxim\\recorded_imgs\\{self.name}_{time.time()}\\"

        # declare some variables
        self.car_loaded = False
        self.aborted = False
        self.iter_image = np.zeros((120, 160, 3))
        self.last_time = time.time()
        self.to_process = {self.last_time: self.iter_image}
        self.crop = 40
        self.previous_st = 0
        self.current_speed = 0
        self.img_buffer = []
        self.cte_history = []
        self.previous_brake = []

    def on_msg_recv(self, json_packet):
        try:
            msg_type = json_packet["msg_type"]

            if msg_type == "car_loaded":
                self.car_loaded = True

            elif msg_type == "aborted":
                self.aborted = True

            elif msg_type == "telemetry":
                imgString = json_packet["image"]
                tmp_img = np.asarray(Image.open(BytesIO(base64.b64decode(imgString))))
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)

                # print(tmp_img.shape)

                self.delay_buffer(tmp_img)
                self.current_speed = json_packet["speed"]

                del json_packet["image"]
                self.last_packet = json_packet

                # if "lidar" in json_packet and json_packet["lidar"] is not None:
                #     print(len(json_packet["lidar"]))

                # if 'pitch' in json_packet:
                #     pitch = json_packet["pitch"]
                #     roll = json_packet["roll"]
                #     yaw = json_packet["yaw"]
                #     print(pitch, roll, yaw)

            else:
                print(json_packet)

        except Exception as e:
            if json_packet != {}:
                print(e)
            pass

    def reset_car(self):
        msg = {"msg_type": "reset_car"}
        self.send_now(json.dumps(msg))

    def get_active_node_coords(self, index=0):
        msg = {"msg_type": "node_position", "index": str(index)}
        self.send_now(json.dumps(msg))

    def move_car(self, x, y, z, qx=None, qy=None, qz=None, qw=None):
        if qx is not None and qy is not None and qz is not None and qw is not None:
            msg = {
                "msg_type": "set_position",
                "pos_x": str(x),
                "pos_y": str(y),
                "pos_z": str(z),
                "Qx": str(qx),
                "Qy": str(qy),
                "Qz": str(qz),
                "Qw": str(qw),
            }
        else:
            msg = {"msg_type": "set_position", "pos_x": str(x), "pos_y": str(y), "pos_z": str(z)}

        self.send_now(json.dumps(msg))

    def delay_buffer(self, img):
        now = time.time()
        self.img_buffer.append((img, now))
        temp_img = None
        to_remove = []
        for it, imgt in enumerate(self.img_buffer):
            if now - imgt[1] > self.buffer_time:
                temp_img = imgt[0]
                temp_time = imgt[1]
                to_remove.append(it)
            else:
                break
        if len(to_remove) > 0:
            self.to_process[temp_time] = temp_img
            for _ in range(len(to_remove)):
                del self.img_buffer[0]
                del to_remove[0]

    def terminate(self):
        if os.path.exists(self.default_dos):
            if len(os.listdir(self.default_dos)) == 0:
                os.rmdir(self.default_dos)
        self.aborted = True

    def load_map(self, track):
        msg = {"msg_type": "load_scene", "scene_name": str(track)}
        self.send_now(json.dumps(msg))

    def get_scene_map(self):
        msg = {"msg_type": "get_scene_names"}
        self.send_now(json.dumps(msg))

    def exit_scene(self):
        msg = {"msg_type": "exit_scene"}
        self.send_now(json.dumps(msg))

    def start(self, custom_body=True, body_msg="", custom_cam=False, cam_msg="", color=[20, 20, 20]):
        if custom_body:  # send custom body info
            if body_msg == "":
                msg = {
                    "msg_type": "car_config",
                    "body_style": "donkey",
                    "body_r": str(color[0]),
                    "body_g": str(color[1]),
                    "body_b": str(color[2]),
                    "car_name": f"Maxime_{self.name}",
                    "font_size": "50",
                }
            else:
                msg = body_msg
            dumped_msg = json.dumps(msg)
            self.send_now(dumped_msg)
            time.sleep(1.0)

        if custom_cam:  # send custom cam info
            if cam_msg == "":
                msg = {
                    "msg_type": "cam_config",
                    "fov": "90",
                    "fish_eye_x": "0.4",
                    "fish_eye_y": "0.7",
                    "img_w": "160",
                    "img_h": "120",
                    "img_d": "4",
                    "img_enc": "PNG",
                    "offset_x": "0.0",
                    "offset_y": "1.120395",
                    "offset_z": "0.5528488",
                    "rot_x": "15.0",
                }
            else:
                msg = cam_msg
            dumped_msg = json.dumps(msg)
            self.send_now(dumped_msg)
            time.sleep(1.0)

        # send lidar settings
        # if self.name == '0':
        #     msg = {"msg_type": "lidar_config",
        #            "degPerSweepInc": "2.0",
        #            "degAngDown": "5",
        #            "degAngDelta": "-1.0",
        #            "numSweepsLevels": "1",
        #            "maxRange": "50.0",
        #            "noise": "0.4",
        #            "offset_x": "0.0",
        #            "offset_y": "0.5",
        #            "offset_z": "0.5",
        #            "rot_x": "0.0"}

        #     self.send_now(json.dumps(msg))
        #     time.sleep(0.2)

    def send_controls(self, steering, throttle, brake):
        p = {"msg_type": "control", "steering": steering.__str__(), "throttle": throttle.__str__(), "brake": brake.__str__()}
        self.send(json.dumps(p))

        # this sleep lets the SDClient thread poll our message and send it out.
        time.sleep(self.poll_socket_sleep_sec)

    def update(self, st, throttle=1.0, brake=0.0):
        self.send_controls(st, throttle, brake)

    def prepare_img(self, img):
        img = img[self.crop:, :, :]
        img = cv2.resize(img, (160, 120))
        return img

    def predict(self, img, transform=True, smooth=True, coef=[-1, -0.5, 0, 0.5, 1], send=False):
        target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings

        img = self.prepare_img(img)
        # to_pred = [np.expand_dims(img, axis=0)/255]

        to_pred = self.dataset.make_to_pred_annotations([img], [[0, self.current_speed]], self.input_components)
        pred, dt = self.model.predict(to_pred)

        if isinstance(pred, list):
            pred = pred[0]

        direction = None
        throttle = None

        if "direction" in pred:
            direction = pred.get("direction", None)
            if isinstance(direction, Iterable):
                direction = direction[0]

        if "throttle" in pred:
            throttle = pred.get("throttle", None)
            if isinstance(throttle, Iterable):
                throttle = throttle[0]

        # assert direction is not None

        if throttle is None:
            throttle = opt_acc(direction, self.current_speed, max_throttle, min_throttle, target_speed)
        else:
            throttle = 1 if throttle > 1 else throttle
            throttle = 0 if throttle < 0 else throttle

        if transform:
            direction = transform_st(direction, sq, mult)

        if send:
            self.update(direction, throttle=throttle, brake=0)
        self.previous_st = direction
        return direction

    def get_keyboard(self, keys=["left", "up", "right"], bkeys=["down"]):
        pressed = []
        bpressed = []
        dpressed = []
        bfactor = 1
        manual = False

        for k in keys:
            pressed.append(keyboard.is_pressed(k))
        keys_st = cat2linear([pressed], coef=[-1, 0, 1], av=True)

        for bk in bkeys:
            bpressed.append(keyboard.is_pressed(bk))
        if any(bpressed):
            bfactor = -1

        if any(pressed + bpressed) and not any(dpressed):
            manual = True

        return manual, keys_st, bfactor

    def get_throttle(self, keys=["c", "d", "e"], bkeys=["down"]):
        pressed = []
        manual = False

        for k in keys:
            pressed.append(keyboard.is_pressed(k))
        keys_th = cat2linear([pressed], coef=[0, 0.5, 1], av=True)

        if any(pressed):
            manual = True

        return manual, keys_th

    def rdm_color_startv1(self, color=[]):
        if color == []:
            color = np.random.randint(0, 255, size=(3))
        self.start(color=color)

    def save_img(self, img, **to_save):
        # print("SAVING")
        tmp_img = self.prepare_img(img)
        self.dataset.save_img_and_annotation(tmp_img, to_save, dos=self.default_dos)

    def get_latest(self):
        if len(self.to_process) > 0:
            return list(self.to_process.items())[-1]
        else:
            return self.last_time, self.iter_image

    def wait_latest(self):
        while len(self.to_process) == 0:
            time.sleep(self.poll_socket_sleep_sec)
        return list(self.to_process.items())[-1]


class universal_client(SimpleClient):
    def __init__(self, params_dict, load_map, name):
        host = params_dict.get("host", "127.0.0.1")
        port = params_dict.get("port", 9091)
        window = params_dict["window"]
        sleep_time = params_dict["sleep_time"]
        PID_settings = params_dict["PID_settings"]
        self.loop_settings = params_dict["loop_settings"]
        buffer_time = params_dict["buffer_time"]
        track = params_dict["track"]
        model = params_dict["model"]
        dataset = params_dict["dataset"]
        input_components = params_dict["input_components"]

        super().__init__(
            (host, port),
            model,
            dataset,
            input_components,
            sleep_time=sleep_time,
            name=name,
            PID_settings=PID_settings,
            buffer_time=buffer_time,
        )
        # self.model._make_predict_function() # useless with tf2

        if load_map:
            self.load_map(track)

        time.sleep(1)
        self.rdm_color_startv1()

        self.t = threading.Thread(target=self.loop)
        self.t.start()

        AutoInterface(window, self)

    def loop(self):
        _, img = self.get_latest()
        self.predict(img, send=False)  # do an init pred
        self.update(0, throttle=0.0, brake=0.1)

        while True:
            try:
                target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings
                transform, smooth, random, do_overide, record, stop = self.loop_settings
                # print(transform, smooth, random, do_overide, record)

                if len(self.to_process) > 0:
                    self.last_time, self.iter_image = self.get_latest()
                else:
                    self.last_time, self.iter_image = self.wait_latest()

                # print(self.last_packet)
                cv2.imshow(self.name, self.prepare_img(self.iter_image))
                cv2.waitKey(1)

                if self.aborted:
                    self.stop()
                    print("stopped client", self.name)
                    break

                if stop:
                    self.update(0, throttle=0.0, brake=1.0)
                    time.sleep(self.poll_socket_sleep_sec)
                    continue

                toogle_manual, manual_st, bk = self.get_keyboard()

                if toogle_manual and do_overide:
                    if transform:
                        manual_st = transform_st(manual_st, sq, mult)

                    manual, throttle = self.get_throttle()
                    if manual is False:
                        throttle = opt_acc(manual_st, self.current_speed, max_throttle, min_throttle, target_speed)

                    if record and len(self.to_process) > 0:
                        self.save_img(
                            self.iter_image,
                            direction=manual_st,
                            speed=self.current_speed,
                            throttle=throttle * bk,
                            time=time.time(),
                        )

                    self.update(manual_st, throttle=throttle * bk)

                else:
                    # print(len(self.to_process)/(time.time() - self.last_time))
                    if len(self.to_process) > 0:
                        self.last_time, self.iter_image = self.get_latest()
                    else:
                        self.last_time, self.iter_image = self.wait_latest()

                    self.predict(self.iter_image, transform=transform, smooth=smooth, send=True)

                # clean up the dict before the next iteration
                # del self.to_process[self.last_time]
                img_times = list(self.to_process.keys())
                for img_time in img_times:
                    if img_time <= self.last_time:
                        del self.to_process[img_time]

            except Exception as e:
                print(f"{self.name}: loop failed to process iteration")
                print(e)


class log_points(SimpleClient):
    def __init__(self, host="127.0.0.1", port=9091, filename="log_points.txt"):
        self.filename = filename
        self.out_file = open(self.filename, "w")
        self.out_file.close()
        self.last_point = None
        self.distance_interval = 2
        self.last_time = time.time()

        super().__init__((host, port), 0, 0, 0)

        time.sleep(1)
        self.rdm_color_startv1()
        self.t = threading.Thread(target=self.loop)
        self.t.start()

    def log(self, px, py, pz):
        # not really efficient way to do this, but it works
        distance = ((px - self.last_point[0]) ** 2 + (py - self.last_point[1]) ** 2 + (pz - self.last_point[2]) ** 2) ** 0.5

        if distance >= self.distance_interval:
            self.out_file = open(self.filename, "a")
            self.out_file.write(f"{px},{py},{pz}\n")
            self.out_file.close()
            self.last_time = time.time()
            self.last_point = (px, py, pz)

    def loop(self):
        self.update(0, throttle=0.0, brake=0.1)
        while True:
            toogle_manual, manual_st, bk = self.get_keyboard()

            if toogle_manual:
                manual, throttle = self.get_throttle()
                self.update(manual_st, throttle=0.3 * bk)

            px = self.last_packet.get("pos_x")
            py = self.last_packet.get("pos_y")
            pz = self.last_packet.get("pos_z")

            if self.last_point is None:
                self.last_point = (px, py, pz)
            elif (px, py, pz) != self.last_point:
                self.log(px, py, pz)


def test_model(dataset: dataset_json.Dataset, input_components, model_path):
    model = architectures.safe_load_model(model_path, compile=False)
    # architectures.apply_predict_decorator(model)

    host = "127.0.0.1"
    port = 9091

    window = windowInterface()  # create a window

    config = {
        "host": host,
        "port": port,
        "window": window,
        "use_speed": (True, True),
        "sleep_time": 0.01,
        "PID_settings": [17, 0.5, 0.3, 1.0, 1.0],
        "loop_settings": [True, False, False, True, False, False],
        "buffer_time": 11,
        "track": "warren",
        "model": model,
        "dataset": dataset,
        "input_components": input_components,
    }

    universal_client(config, True, "model_test")
    window.mainloop()


if __name__ == "__main__":
    model_path = os.getcwd() + os.path.normpath("/test_model/models/working_epita3.tflite")
    model = architectures.safe_load_model(model_path, output_names=["direction"])
    # model.summary()

    dataset = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
    input_components = []

    hosts = ["127.0.0.1", "donkey-sim.roboticist.dev", "sim.diyrobocars.fr"]
    host = hosts[0]
    port = 9091

    window = windowInterface()  # create a window

    config = {
        "host": host,
        "port": port,
        "window": window,
        "use_speed": (True, True),
        "sleep_time": 0.01,
        "PID_settings": [17, 0.5, 0.3, 1.1, 1.0],
        "loop_settings": [True, False, False, True, False, False],
        "buffer_time": 0,
        "track": "circuit_launch",
        "model": model,
        "dataset": dataset,
        "input_components": input_components,
    }

    load_map = True
    client_number = 1
    for i in range(client_number):
        universal_client(config, load_map, str(i))
        print("started client", i)
        load_map = False

    window.mainloop()  # only display when everything is loaded to avoid threads to be blocked !
