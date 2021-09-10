import cv2

from custom_modules.datasets import dataset_json

if __name__ == "__main__":
    Dataset = dataset_json.Dataset(["direction", "time"])
    paths = Dataset.load_dos_sorted("C:\\Users\\maxim\\random_data\\ironcar\\ironcar\\")

    for path in paths:
        img, annotation = Dataset.load_img_and_annotation(path)

        img = cv2.resize(img, (480, 360))
        cv2.imshow("img", img)

        cv2.waitKey(1)
