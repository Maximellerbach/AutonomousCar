from customDataset import DatasetJson
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":

    datasetJson = DatasetJson(['direction', 'speed', 'throttle', 'time'])

    # src_dos = "C:\\Users\\maxim\\random_data\\20 checkpoint patch"
    dst_dos = "C:\\Users\\maxim\\random_data\\test\\"

    dos_name = '20 checkpoint patch'

    for json_path in tqdm(glob(f'{dst_dos}\\{dos_name}'+'\\*.json')):
        annotations = datasetJson.load_annotation(json_path, to_list=False)
        annotations['dos'] = dst_dos+dos_name+"\\"
        annotations['img_path'] = annotations['dos']+str(annotations['time'])+".png"
        datasetJson.save_annotations_dict(annotations)
