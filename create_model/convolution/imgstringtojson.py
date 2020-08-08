import dataset
from customDataset import DatasetJson, direction_component, speed_component, throttle_component, time_component
from glob import glob

if __name__ == "__main__":

    dataset = dataset.Dataset(
        [dataset.direction_component, dataset.speed_component, dataset.throttle_component, dataset.time_component])
    datasetJson = DatasetJson([direction_component, speed_component, throttle_component, time_component])

    src_dos = "C:\\Users\\maxim\\random_data\\19 custom speed"
    dst_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"

    dosdir = False

    if dosdir:
        doss = glob(src_dos+"*")

        for dos in doss:
            dos_name = dos.split('\\')[-1]
            datasetJson.imgstring2json_dos(
                dataset, datasetJson.imgstring2json, dos+"\\", dst_dos+dos_name+"\\")
    else:
        dos_name = src_dos.split('\\')[-1]
        datasetJson.imgstring2json_dos(
            dataset, datasetJson.imgstring2json, src_dos+"\\", dst_dos+dos_name+"\\")
