from glob import glob

from tqdm import tqdm

from custom_modules.datasets import dataset_json

if __name__ == "__main__":
    import os
    base_path = os.getenv('ONEDRIVE') + "\\random_data"

    datasetJson = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'time'])

    dst_dos = f"{base_path}\\test\\20 checkpoint patch"

    for json_path in tqdm(glob(f'{dst_dos}\\*.json')):
        annotation = datasetJson.load_annotation(json_path, to_list=False)
        annotation['dos'] = dst_dos+"\\"
        annotation['img_path'] = annotation['dos'] + \
            str(annotation['time'])+".png"
        datasetJson.save_annotation_dict(annotation)
