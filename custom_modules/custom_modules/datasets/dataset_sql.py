import base64
import time

import cv2
import numpy as np
from PIL import Image

from ..psql import db_utils, queries


class Dataset():
    """Dataset class that contains everything needed to load and save a sql dataset."""

    def __init__(self):
        db_utils.start_if_not_running()
        self.conn = db_utils.connect()

    def encode_image(self, image):
        return base64.b64encode(image)

    def decode_image(self, image):
        bytes_img = base64.b64decode(image)
        return np.frombuffer(bytes_img, dtype=np.int8)

    def save_image_dict(self, image_dict):
        if 'time' not in image_dict.keys():
            image_dict['time'] = time.time()

        # convert numpy array image to base64 encoded string
        if not isinstance(image_dict['image'], str):
            image_dict['image'] = self.encode_image(image_dict['image'])

        queries.image.add(conn, image_dict)

    def save_annotation_dict(self, annotation):
        queries.label.add(annotation)

    def save_img_and_annotation(self, img, annotation, dos=None):
