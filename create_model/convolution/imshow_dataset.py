import dataset
import cv2
import numpy as np

Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])

gdos = Dataset.load_dataset_sequence("C:\\Users\\maxim\\datasets\\", max_interval=0.2)
print(len(gdos))

prev_time = 0

for dos in gdos:
    for sequences in dos:
        for it, s in enumerate(sequences):
            for impath in s:
                img = cv2.imread(impath)/255

                time = Dataset.load_component_item(impath, -1)
                dt = time-prev_time 
                if dt>0.2:
                    print(dt)
                prev_time = time

                cv2.imshow(str(it)+'img', img)
                cv2.waitKey(1)