import dataset
from customDataset import DatasetJson, direction_component, speed_component, throttle_component, time_component
import os

if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob

    datasetJson = DatasetJson([direction_component, speed_component, throttle_component, time_component])

    # src_dos = "C:\\Users\\maxim\\random_data\\20 checkpoint patch"
    dst_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"

    dos_name = '20 checkpoint patch'

    for json_path in tqdm(glob(dst_dos+'\\20 checkpoint patch'+'\\*.json')):
        annotations = datasetJson.load_annotation(json_path, to_list=False)
        annotations['dos'] = dst_dos+dos_name+"\\"
        annotations['img_path'] = annotations['dos']+annotations['time']+".png"
        datasetJson.save_annotations_dict(annotations)
