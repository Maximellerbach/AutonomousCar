import base64
import io
import json
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image


class DatasetJson():
    def __init__(self, lab_structure):
        self.label_structure = [i() for i in lab_structure]
        self.other_components = [img_path_component()]
        self.format = '.json'

    def save_img_and_annotations(self, dos, img, annotations):
        assert(len(annotations) == len(self.label_structure))

        to_save_dict = {'dos': dos}
        for component, annotation in zip(self.label_structure, annotations):
            to_save_dict = component.add_item_to_dict(annotation, to_save_dict)

        for component in self.other_components:
            to_save_dict = component.add_item_to_dict(to_save_dict)

        time = to_save_dict['time']
        cv2.imwrite(to_save_dict['img_path'], img)
        with open(os.path.normpath(f'{dos}{time}{self.format}'), 'w') as json_file:
            json.dump(to_save_dict, json_file)

    def save_img_and_annotations_deprecated(self, dos, img, annotations):
        path = dos+str(annotations[-1])  # save json with time component
        annotations.insert(-1, path+'.png')  # add the img_path component
        cv2.imwrite(annotations[-2], img)

        annotations_dict = self.annotations_to_dict_deprecated(annotations)
        with open(path+self.format, 'w') as json_file:
            json.dump(annotations_dict, json_file)

    def save_img_encoded_json_deprecated(self, dos, imgpath, annotations):
        path = dos+str(annotations[-1])
        annotations.insert(-1, self.label_structure[-2].encode_img(imgpath))

        annotations_dict = self.annotations_to_dict_deprecated(annotations)
        with open(path+self.format, 'w') as json_file:
            json.dump(annotations_dict, json_file)

    def annotations_to_dict_deprecated(self, annotations):
        annotations_dict = {}
        for it, component in enumerate(self.label_structure):
            annotations_dict[component.name] = annotations[it]

        return annotations_dict

    def load_json(self, path):
        with open(path, "r") as json_file:
            json_data = json.load(json_file)  # json -> dict
        return json_data

    def load_component_item(self, path, n_component):
        json_data = self.load_json(path)
        return self.label_structure[n_component].get_item(json_data)

    def load_annotation(self, path):
        def get_item_comp(component):
            return component.get_item(json_data)
        json_data = self.load_json(path)
        annotations = list(map(get_item_comp, self.label_structure))

        return annotations

    def load_img_and_annotation(self, path):
        annotations = self.load_annotation(path)
        img = cv2.imread(path.replace('.json', '.png'))

        return img, annotations

    def sort_paths_by_component(self, paths, n_component):
        def sort_function(path):
            return self.load_component_item(path, n_component)

        paths = sorted(paths, key=sort_function)
        return paths

    def split_sorted_paths(self, paths, time_interval=0.2):
        splitted = [[]]
        n_split = 0

        for i in range(1, len(paths)):
            prev = paths[i-1]
            actual = paths[i]

            prev_t = self.load_component_item(prev, -1)
            actual_t = self.load_component_item(actual, -1)

            dt = actual_t-prev_t
            if abs(dt) > time_interval:
                n_split += 1
                splitted.append([])

            splitted[n_split].append(paths[i])

        return splitted

    def save_json_sorted(self, sorted, path):
        save_dict = dict(enumerate(sorted))
        with open(path, 'w') as json_file:
            json.dump(save_dict, json_file)

    def load_dos(self, dos):
        return glob(dos+"*.json")

    def load_dos_sorted(self, dos):
        json_dos_name = dos[:-1]
        json_path = f'{json_dos_name}{self.format}'
        if os.path.isfile(json_dos_name):
            sorted_path_json = self.load_json(json_path)

            if len(sorted_path_json) == len(os.listdir(dos))//2:
                sorted_paths = [sorted_path_json[i] for i in sorted_path_json]
                return sorted_paths

        new_sorted_paths = self.sort_paths_by_component(self.load_dos(dos), -1)
        self.save_json_sorted(new_sorted_paths, json_path)
        return new_sorted_paths

    def load_dataset(self, doss):
        doss_paths = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos(dos+"\\")
                doss_paths.append(paths)

        return doss_paths

    def load_dataset_sorted(self, doss):
        doss_paths = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                doss_paths.append(paths)

        return doss_paths

    def load_dataset_sequence(self, doss, max_interval=0.2):
        doss_sequences = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                paths_sequence = self.split_sorted_paths(
                    paths, time_interval=max_interval)
                doss_sequences.append(list(paths_sequence))

        return doss_sequences

    def imgstring2json(self, dataset_obj, dst_dos, path):
        img = dataset_obj.load_image(path)
        annotations = dataset_obj.load_annotation(path)
        self.save_img_and_annotations(dst_dos, img, annotations)

    def imgstring2json_encoded_deprecated(self, dataset_obj, dst_dos, path):
        annotations = dataset_obj.load_annotation(path)
        self.save_img_encoded_json_deprecated(dst_dos, path, annotations)

    def imgstring2json_dos(self, dataset_obj, imgstring2dos_function, src_dos, dst_dos):
        from tqdm import tqdm
        try:
            os.mkdir(dst_dos)
        except:
            pass
        paths = dataset_obj.load_dos(src_dos)
        for path in tqdm(paths):
            imgstring2dos_function(dataset_obj, dst_dos, path)


class direction_component:
    def __init__(self):
        self.name = "direction"
        self.type = float
        self.do_flip = True

    def get_item(self, json_data):
        return float(json_data[self.name])

    def add_item_to_dict(self, item, annotations_dict):
        if isinstance(item, self.type):
            annotations_dict[self.name] = item
        else:
            ValueError(f'item type: {type(item)} should match {self.type}')
        return annotations_dict


class speed_component:
    def __init__(self):
        self.name = "speed"
        self.type = float
        self.do_flip = False

    def get_item(self, json_data):
        return self.type(json_data[self.name])

    def add_item_to_dict(self, item, annotations_dict):
        if isinstance(item, self.type):
            annotations_dict[self.name] = item
        else:
            ValueError(f'item type: {type(item)} should match {self.type}')
        return annotations_dict


class throttle_component:
    def __init__(self):
        self.name = "throttle"
        self.type = float
        self.do_flip = False

    def get_item(self, json_data):
        return self.type(json_data[self.name])

    def add_item_to_dict(self, item, annotations_dict):
        if isinstance(item, self.type):
            annotations_dict[self.name] = item
        else:
            ValueError(f'item type: {type(item)} should match {self.type}')
        return annotations_dict


class time_component:
    def __init__(self):
        self.name = "time"
        self.type = float
        self.do_flip = False

    def get_item(self, json_data):
        return float(json_data[self.name])

    def add_item_to_dict(self, item, annotations_dict):
        if isinstance(item, self.type):
            annotations_dict[self.name] = item
        else:
            ValueError(f'item type: {type(item)} should match {self.type}')
        return annotations_dict


class img_path_component:
    def __init__(self):
        self.name = "img_path"
        self.type = str
        self.do_flip = False

    def get_item(self, json_data):
        return self.type(json_data[self.name])

    def add_item_to_dict(self, annotations_dict):
        time = annotations_dict['time']
        dos = annotations_dict['dos']
        img_path = os.path.normpath(f'{dos}/{time}.png')

        annotations_dict[self.name] = img_path
        return annotations_dict


class imgbase64_component:
    def __init__(self):
        self.name = "img_base64"
        self.type = str
        self.do_flip = False

    def get_item(self, json_data):
        # getting the image from string and convert it to BGR
        base64_img = self.type(json_data[self.name])
        base64_img_decoded = base64.b64decode(base64_img)
        return cv2.cvtColor(np.array(Image.open(io.BytesIO(base64_img_decoded))), cv2.COLOR_RGB2BGR)

    def encode_img(self, imgpath):
        with open(imgpath, "rb") as img_file:
            img_encoded = base64.b64encode(img_file.read()).decode('utf-8')
        return img_encoded

    def add_item_to_dict(self, img_path, annotations_dict):
        encoded = self.encode_img(img_path)
        annotations_dict[self.name] = encoded

        return annotations_dict


if __name__ == "__main__":
    # quick code to test image decoding from string
    dataset_json = DatasetJson(
        [direction_component, img_path_component, time_component])

    dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"
    gdos = dataset_json.load_dataset_sorted(dos)
    for dos in gdos:
        for path in dos:
            img, annotations = dataset_json.load_img_and_annotation(path)
            cv2.imshow('img', img)
            cv2.waitKey(1)
