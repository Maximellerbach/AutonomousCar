from glob import glob
import os

from tqdm import tqdm

from custom_modules.datasets import dataset_json


def assemble_string(str_list, sep='\\'):
    s = ''
    for string in str_list:
        s += string + sep
    return s


def replace_string(path: str, string: str = '-Maxime-PC', replace_string: str = ''):
    return path.replace(string, replace_string)


def merge_changes(base_path: str):
    # use only when there are changes conflicts with onedrive
    for json_path in tqdm(glob(f"{base_path}\\**\\*.json", recursive=True)):
        if '-Maxime-PC' in json_path:
            new_path = replace_string(json_path)
            os.remove(new_path)
            os.rename(json_path, new_path)


def change_directory(base_path: str, datasetJson: dataset_json.Dataset):
    for json_path in tqdm(glob(f"{base_path}\\**\\*.json", recursive=True)):

        if os.path.isdir(assemble_string(json_path.split('.json')[:-1])):
            continue

        annotation = datasetJson.load_annotation(json_path, to_list=False)
        annotation['dos'] = assemble_string(json_path.split('\\')[:-1])
        annotation['img_path'] = annotation['dos'] + \
            str(annotation['time'])+".png"
        datasetJson.save_annotation_dict(annotation)


if __name__ == "__main__":
    base_path = os.getenv('ONEDRIVE') + "\\random_data\\recorded_imgs\\0_1600805893.911268\\"

    datasetJson = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'time'])

    # change_directory(base_path, datasetJson)
    merge_changes(base_path)
