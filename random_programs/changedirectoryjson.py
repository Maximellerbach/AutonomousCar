from glob import glob

from tqdm import tqdm

from custom_modules.datasets.dataset_json import Dataset as DatasetJson

if __name__ == "__main__":

    datasetJson = DatasetJson(['direction', 'speed', 'throttle', 'time'])

    # src_dos = "C:\\Users\\maxim\\random_data\\20 checkpoint patch"
    dst_dos = "C:\\Users\\maxim\\random_data\\test\\"

    dos_name = '20 checkpoint patch'

    for json_path in tqdm(glob(f'{dst_dos}\\{dos_name}'+'\\*.json')):
        annotation = datasetJson.load_annotation(json_path, to_list=False)
        annotation['dos'] = dst_dos+dos_name+"\\"
        annotation['img_path'] = annotation['dos']+str(annotation['time'])+".png"
        datasetJson.save_annotation_dict(annotation)
