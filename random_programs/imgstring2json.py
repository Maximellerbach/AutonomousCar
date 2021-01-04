from custom_modules.datasets import dataset_img, dataset_json
from glob import glob
import os
import cv2
import time
import json
from tqdm import tqdm


def imgstring2donkeyjson_dos(src_dos, dst_dos):
    try:  # ugly way to create a dir if not exist
        os.mkdir(dst_dos)
    except OSError:
        pass

    paths = glob(src_dos+"*")
    for path in tqdm(paths):
        img = cv2.imread(path)
        split_path = path.split('\\')[-1].split('_')
        direction = (int(split_path[0])-7)/4
        new_name = f'{time.time()}'
        new_img_name = new_name+'.png'

        annotation_dict = {
            'cam/image_array': new_img_name,
            'user/angle': direction,
            'user/throttle': 0.5
        }
        # print(annotation_dict)

        with open(dst_dos+new_name+'.json', 'w') as json_file:
            json.dump(annotation_dict, json_file)

        cv2.imwrite(dst_dos+new_img_name, img)
        time.sleep(0.0001)


def imgstring2json_dos(src_Dataset, dst_Dataset, src_dos, dst_dos):
    dos_name = src_dos.split('\\')[-1]
    DatasetJson.imgstring2json_dos(
        DatasetImg, DatasetJson.imgstring2dict2json, src_dos+"\\", dst_dos+dos_name+"\\")


if __name__ == "__main__":

    DatasetImg = dataset_img.Dataset(
        [dataset_img.direction_component, dataset_img.time_component])
    DatasetJson = dataset_json.Dataset(
        ["direction", "speed", "throttle", "time"])
    direction_cmp = DatasetJson.get_component("direction")
    direction_cmp.offset = -7
    direction_cmp.scale = 1/4

    src_doss = [
        "C:\\Users\\maxim\\random_data\\1 ironcar driving\\"
    ]
    dst_dos = "C:\\Users\\maxim\\random_data\\ironcar\\ironcar\\"

    for src_dos in src_doss:
        # imgstring2donkeyjson_dos(src_dos, dst_dos)
        imgstring2json_dos(DatasetImg, DatasetJson, src_dos, dst_dos)
