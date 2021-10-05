from threading import Thread

import cv2
import time


class usbWebcam():
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)

        self.last_image = None
        self.new_image = False

        self.running = True

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            ret, img = self.cap.read()
            if ret is False:
                self.running = False
            else:
                self.last_image = img
                self.new_image = True

    def read(self):
        if self.running:
            if self.new_image:
                self.new_image = False
                return self.last_image
            else:
                while self.new_image is False and self.running:
                    time.sleep(0.001)

                self.new_image = False
                return self.last_image
        else:
            raise Exception("Is the camera connected?")

    def release(self):
        self.running = False
        self.cap.release()
