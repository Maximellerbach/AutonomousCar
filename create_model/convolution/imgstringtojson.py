import dataset
import dataset_json
import os

if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob

    Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])
    DatasetJson = dataset_json.DatasetJson([dataset_json.direction_component, dataset_json.img_name_component, dataset_json.time_component])

    src_dos = "C:\\Users\\maxim\\datasets\\"
    dst_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"
    doss = glob(src_dos+"*")

    for dos in doss:
        dos_name = dos.split('\\')[-1]
        DatasetJson.imgstring2json_dos(Dataset, DatasetJson.imgstring2json, dos+"\\", dst_dos+dos_name+"\\")

