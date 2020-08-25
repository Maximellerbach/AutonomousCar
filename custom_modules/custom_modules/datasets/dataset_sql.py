import base64
import time

import cv2
import numpy as np

from ..psql import db_utils
from ..psql.queries import label


class Dataset():
    """Dataset class that contains everything needed to load and save a sql dataset, not used for the moment."""

    def __init__(self, lab_structure):
        db_utils.start_if_not_running()
        self.conn = db_utils.connect()
        self.rows_meta = label.fetch_rows_meta(self.conn)

    def save_annotation_dict(self, annotation_dict, dataset_name=None):
        if dataset_name is not None:
            annotation_dict['dataset_name'] = dataset_name
        return label.add(self.conn, annotation_dict)

    def load_image(self, img_path):
        return cv2.imread(img_path)

    def load_annotation_by_id(self, item_id):
        return label.fetch_by_id(self.conn, item_id)

    def load_image_and_annotations_by_id(self, item_id):
        annotation = self.load_annotation(item_id)
        img_path = annotation.get('image_path')

        return (
            self.load_image(img_path),
            annotation
        )

    def load_dataset_sorted(self, dataset_name):
        return list(label.generator_load_dataset(self.conn, dataset_name))

    def load_sorted(self, dataset_name):
        return list(label.generator_load(self.conn, dataset_name))
