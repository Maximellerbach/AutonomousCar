from custom_modules.datasets import dataset, dataset_json
from glob import glob

dataset = dataset.Dataset(
    [dataset.direction_component, dataset.time_component])
datasetJson = dataset_json.DatasetJson(
    ["direction", "speed", "throttle", "time"])

src_dos = "C:\\Users\\maxim\\random_data\\12 sim circuit 2 new"
dst_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"
dosdir = False

if dosdir:
    doss = glob(src_dos+"*")
    for dos in doss:
        dos_name = dos.split('\\')[-1]
        datasetJson.imgstring2json_dos(
            dataset, datasetJson.imgstring2dict2json, dos+"\\", dst_dos+dos_name+"\\")
else:
    dos_name = src_dos.split('\\')[-1]
    datasetJson.imgstring2json_dos(
        dataset, datasetJson.imgstring2dict2json, src_dos+"\\", dst_dos+dos_name+"\\")
