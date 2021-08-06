import base64
import os
import threading
import time
from io import BytesIO

import cv2
import numpy as np
import quaternion
from custom_modules.datasets import dataset_json
from PIL import Image

import sim_client


class auto_labeling(sim_client.SimpleClient):
    def __init__(self, dataset, host='127.0.0.1', port=9091):
        super().__init__((host, port), "", dataset, "")
        self.n_lateral = 2
        self.road_width = 2
        self.width_offset = 0
        self.height_offset = 0.0

        self.n_rot_y = 3
        self.max_rot = 30/180

        self.pos_offset_values = [
            (i/self.n_lateral) * self.road_width for i in range(-self.n_lateral, self.n_lateral+1)]
        print(self.pos_offset_values)

        self.rot_offset_values = [
            (i/self.n_rot_y) * self.max_rot for i in range(-self.n_rot_y, self.n_rot_y+1)]
        print(self.rot_offset_values)

        # steering controller factor
        self.kp = 1.25
        # self.kd = 1

        self.node_info = {}
        self.got_node_coords = False
        self.got_telemetry = False
        self.image = np.zeros((120, 160, 3))

        os.mkdir(self.default_dos)

        time.sleep(1)
        self.rdm_color_startv1()
        self.t = threading.Thread(target=self.loop)
        self.t.start()

    def on_msg_recv(self, json_packet):
        try:
            msg_type = json_packet['msg_type']
            if msg_type == "node_position":
                self.node_info = json_packet
                self.got_node_coords = True

            elif msg_type == "telemetry":
                if self.got_telemetry is False:
                    imgString = json_packet["image"]
                    tmp_img = np.asarray(Image.open(
                        BytesIO(base64.b64decode(imgString))))
                    self.image = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)
                    self.got_telemetry = True

            else:
                print(json_packet)

        except:
            if json_packet != {}:
                print(json_packet)

    def await_get_active_node_coords(self, index):
        self.get_active_node_coords(index)

        while(not self.got_node_coords):
            pass

        self.got_node_coords = False

    def await_telemetry(self):
        while(not self.got_telemetry):
            pass

        self.got_telemetry = False

    def get_steering(self, car_pos, car_rot, target_pos, target_rot):
        relative_pos = target_pos - car_pos
        distance = np.linalg.norm(relative_pos)
        no_rot_relative_pos = quaternion.rotate_vectors(1/car_rot, relative_pos)/distance

        x_difference = no_rot_relative_pos[0]

        steering = np.clip(x_difference*self.kp, -1.0, 1.0)
        return steering

    def loop(self):
        self.update(0, throttle=0.0, brake=0.1)

        index = 100
        while(True):
            self.await_get_active_node_coords(index)
            node_rotation = np.quaternion(self.node_info.get('Qx'),
                                          self.node_info.get('Qy'),
                                          self.node_info.get('Qz'),
                                          self.node_info.get('Qw'))
            node_position = np.array([self.node_info['pos_x'],
                                      self.node_info['pos_y'] + self.height_offset,
                                      self.node_info['pos_z']])

            # point ahead
            self.await_get_active_node_coords(index + 2)
            target_node_rotation = np.quaternion(self.node_info.get('Qx'),
                                                 self.node_info.get('Qy'),
                                                 self.node_info.get('Qz'),
                                                 self.node_info.get('Qw'))
            target_node_position = np.array([self.node_info['pos_x'],
                                             self.node_info['pos_y'] + self.height_offset,
                                             self.node_info['pos_z']])

            for pos_x_offset in self.pos_offset_values:
                offset_pos = np.array([pos_x_offset + self.width_offset, 0, 0])
                offset_pos = quaternion.rotate_vectors(
                    node_rotation, offset_pos)
                car_pos = offset_pos + node_position

                for rot_y_offset in self.rot_offset_values:
                    rot_y_offset_quaternion = np.quaternion(0, rot_y_offset, 0, 1)
                    car_rot = node_rotation * rot_y_offset_quaternion

                    self.move_car(car_pos[0],
                                  car_pos[1],
                                  car_pos[2],
                                  qx=car_rot.x,
                                  qy=car_rot.y,
                                  qz=car_rot.z,
                                  qw=car_rot.w)

                    # wait for the msg to get to the server and to be processed
                    time.sleep(1/40)
                    self.got_telemetry = False

                    self.await_telemetry()
                    cv2.imshow('image', self.image)
                    cv2.waitKey(1)

                    steering = self.get_steering(car_pos, car_rot, target_node_position, target_node_rotation)
                    # print(steering)

                    self.save_img(self.image, direction=steering,
                                  time=time.time())

            index += 1  # you can skip some nodes


if __name__ == '__main__':
    dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'time'])

    hosts = ['127.0.0.1', 'donkey-sim.roboticist.dev', 'sim.diyrobocars.fr']
    host = hosts[0]
    port = 9091

    auto_labeling(dataset, host, port)
