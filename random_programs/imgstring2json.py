from custom_modules.datasets import dataset_img, dataset_json
from glob import glob

DatasetImg = dataset_img.Dataset(
    [dataset_img.direction_component, dataset_img.time_component])
DatasetJson = dataset_json.Dataset(
    ["direction", "speed", "throttle", "time"])

src_dos = "D:\\Maxime\\OneDrive\\random_data\\2 donkeycar driving"
dst_dos = "D:\\Maxime\\OneDrive\\random_data\\json_dataset\\"
dosdir = False

if dosdir:
    doss = glob(src_dos+"*")
    for dos in doss:
        dos_name = dos.split('\\')[-1]
        DatasetJson.imgstring2json_dos(
            DatasetImg, DatasetJson.imgstring2dict2json, dos+"\\", dst_dos+dos_name+"\\")
else:
    dos_name = src_dos.split('\\')[-1]
    DatasetJson.imgstring2json_dos(
        DatasetImg, DatasetJson.imgstring2dict2json, src_dos+"\\", dst_dos+dos_name+"\\")
