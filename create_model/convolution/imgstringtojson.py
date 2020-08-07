import dataset
from customDataset import DatasetJson, direction_component, time_component
import os

if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob

    dataset = dataset.Dataset(
        [dataset.direction_component, dataset.time_component])
    datasetJson = DatasetJson([direction_component, time_component])

    src_dos = "C:\\Users\\maxim\\datasets\\"
    dst_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"
    doss = glob(src_dos+"*")

    for dos in doss:
        dos_name = dos.split('\\')[-1]
        datasetJson.imgstring2json_dos(
            dataset, datasetJson.imgstring2json, dos+"\\", dst_dos+dos_name+"\\")
