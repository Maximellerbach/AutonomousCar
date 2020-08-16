import datetime
import os

import cv2
import numpy as np


class camera:
    "take a picture and saved it in the static folder"

    def __init__(self):
        # take the first camera
        self.cam = cv2.VideoCapture(0)
        pass

    def TakePicture(self):
        # read a frame
        ret, frame = self.cam.read()
        # where the code is running should be /app
        img_name = os.getcwd() + os.path.normpath("/static/image.jpg")
        cv2.imwrite(img_name, frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # close camera
        self.cam.release()
