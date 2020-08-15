from customDataset import DatasetJson
import cv2
import numpy as np


Dataset = DatasetJson(["direction", "speed", "throttle", "time"])
gdos = Dataset.load_dataset_sequence(
    "C:\\Users\\maxim\\random_data\\json_dataset\\",
    max_interval=0.2)

sequences = np.concatenate([i for i in gdos])
np.random.shuffle(sequences)

for sequence in sequences:
    for labpath in sequence:
        img, annotations = Dataset.load_img_and_annotation(labpath)
        print(annotations)

        cv2.imshow('img', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
