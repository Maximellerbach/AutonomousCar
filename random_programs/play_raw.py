import cv2

from custom_modules.datasets import dataset_json

if __name__ == "__main__":
    Dataset = dataset_json.Dataset(['direction', 'time'])
    paths = Dataset.load_dos_sorted('C:\\Users\\maxim\\image_raw\\')

    for path in paths:
        img, annotation = Dataset.load_img_and_annotation(path)

        cv2.imshow('img', img)

        cv2.waitKey(1)
