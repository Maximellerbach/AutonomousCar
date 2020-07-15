import dataset
import cv2
import numpy as np

Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])
gdos = Dataset.load_dataset_sequence("C:\\Users\\maxim\\datasets\\", max_interval=0.2)
gdos = np.concatenate([i for i in gdos])
np.random.shuffle(gdos)

for sequences in gdos:
    for impath in sequences:
        img = cv2.imread(impath)/255

        cv2.imshow('img', img)
        k = cv2.waitKey(1)
        if k == 27:
            break