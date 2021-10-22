import os
import cv2

from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_lab


def crop(img, top=0.0, bot=0.0, h=120):
    return img[int(top*h):h-int(h*bot)].copy()


Dataset = dataset_json.Dataset(["direction", "speed", "time"])
base_path = os.path.expanduser("~") + "\\random_data"
data_path = f"{base_path}\\rbrl3\\"

paths = Dataset.load_dataset_sorted(data_path, flat=True)
print(len(paths))


for path in paths:
    img, annotation = Dataset.load_img_and_annotation(path, to_list=False)

    # crop the image and resize it
    cropped = crop(img, top=0.1, bot=0.1)
    cropped = cv2.resize(cropped, (160, 120))
    cv2.imshow('img', img)
    cv2.imshow('cropped', cropped)
    cv2.waitKey(1)

    # save the image and annotion somewhere else
    annotation["dos"] = f"{base_path}\\testcrop\\"
    Dataset.save_img_and_annotation(cropped, annotation)
