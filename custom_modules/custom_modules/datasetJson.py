import base64
import io
import json
import os
import time
from glob import glob

import cv2
import numpy as np
from PIL import Image


class DatasetJson():
    """Dataset class that contains everything needed to load and save a json dataset."""

    def __init__(self, lab_structure):
        """Load the given label structure

        Args:
            lab_structure (list): list of components (class)
        """
        self.set_label_structure(lab_structure)  # you could set label_structure again
        self.__meta_components = [img_path_component()]  # this is not changing

        self.components_name_mapping = dict(
            [(i.name, i) for i in self.__label_structure])
        self.format = '.json'

    def set_label_structure(self, lab_structure):
        """Set the label structure to the given lab structure.

        Args:
            lab_structure (list): list of components(class) or list of string
        """
        if isinstance(lab_structure[0], str):
            self.__label_structure = self.names2components(lab_structure)
        else:
            self.__label_structure = [i() for i in lab_structure]

    def save_annotations_dict(self, annotations):
        """Save the annotations dict to {dos}{time}{self.format}.

        Args:
            annotations (dict): dict containing annotations
        """
        dos = annotations.get('dos')
        time_cmp = annotations.get('time')
        with open(os.path.normpath(f'{dos}{time_cmp}{self.format}'), 'w') as json_file:
            json.dump(annotations, json_file)

    def save_img_and_annotations(self, img, annotations, dos=None):
        """Save an image with it's annotations.

        Args:
            img (np.ndarray): image to be saved
            annotations (dict): dict containing annotations to be saved
            dos (string, optional): dos component if not set previously. Defaults to None.
        """

        if isinstance(annotations, list):
            if dos is None:
                raise "dos keyword should be specified"
            assert(len(annotations) == len(self.__label_structure))

            to_save_dict = {'dos': dos}
            for component, annotation in zip(self.__label_structure, annotations):
                to_save_dict = component.add_item_to_dict(
                    annotation, to_save_dict)

        elif isinstance(annotations, dict):
            # check if there is the expected number of labels
            if len(annotations.keys()) != len(self.__label_structure):
                to_save_dict = {}
                for component in self.__label_structure:
                    to_save_dict[component.name] = component.get_item(
                        annotations)
            else:
                to_save_dict = annotations

            if 'dos' not in to_save_dict.keys() and dos is not None:
                to_save_dict["dos"] = dos
            else:
                raise ValueError('dos keyword should be specified')

        else:
            raise ValueError(
                f'annotations should be a list or a dict, not {type(annotations)}')

        for component in self.__meta_components:
            to_save_dict = component.add_item_to_dict(to_save_dict)

        time_cmp = to_save_dict.get('time', time.time())
        cv2.imwrite(to_save_dict.get('img_path', ""), img)
        with open(os.path.normpath(f'{dos}{time_cmp}{self.format}'), 'w') as json_file:
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
        component = self.get_component('img_base64')
        annotations.insert(-1, component.encode_img(imgpath))

        annotations_dict = self.annotations_to_dict_deprecated(annotations)
        with open(path+self.format, 'w') as json_file:
            json.dump(annotations_dict, json_file)

    def annotations_to_dict_deprecated(self, annotations):
        annotations_dict = {}
        for it, component in enumerate(self.__label_structure):
            annotations_dict[component.name] = annotations[it]

        return annotations_dict

    def load_json(self, path):
        """Load a json json_file.

        Args:
            path (string): path to the file

        Returns:
            dict: json file's content
        """
        with open(path, "r") as json_file:
            json_data = json.load(json_file)  # json -> dict
        return json_data

    def get_component(self, n_component):
        if isinstance(n_component, int):
            return self.__label_structure[n_component]
        elif isinstance(n_component, str):
            return self.components_name_mapping[n_component]
        else:
            raise TypeError("n_component must be an integer or a string")

    def load_component_item(self, path, n_component):
        """Load the given component's item from a given json file.

        Args:
            path (string): path to the json file
            n_component (int, string): index/name of the component

        Returns:
            variable: the loaded item
        """
        json_data = self.load_json(path)
        return self.get_component(n_component).get_item(json_data)

    def load_annotation(self, path, to_list=True):
        def get_item_comp(component):
            return component.get_item(json_data)
        json_data = self.load_json(path)
        if to_list:
            return list(map(get_item_comp, self.__label_structure))
        else:
            annotation_dict = {}
            for component in self.__label_structure:
                annotation_dict[component.name] = get_item_comp(component)
            return annotation_dict

    def load_meta(self, path, to_list=True):
        def get_item_comp(component):
            return component.get_item(json_data)
        json_data = self.load_json(path)
        if to_list:
            return list(map(get_item_comp, self.__meta_components))
        else:
            meta_dict = {}
            for component in self.__meta_components:
                meta_dict[component.name] = get_item_comp(component)
            return meta_dict

    def load_img_and_annotation(self, path, to_list=True):
        annotations = self.load_annotation(path, to_list=to_list)
        img_path = self.load_meta(path, to_list=False).get('img_path')
        if img_path is None:
            raise ValueError('json does not have "img_path" component')
        img = cv2.imread(img_path)
        return img, annotations

    def load_img(self, path):
        img_path = self.load_meta(path, to_list=False).get('img_path')
        return cv2.imread(img_path)

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

    def generator_split_sorted_paths(self, paths, time_interval=0.2):
        splitted = []
        for i in range(1, len(paths)):
            prev = paths[i-1]
            actual = paths[i]

            prev_t = self.load_component_item(prev, -1)
            actual_t = self.load_component_item(actual, -1)

            dt = actual_t-prev_t
            if abs(dt) > time_interval:
                yield splitted
                splitted = []
            else:
                splitted.append(paths[i])

    def save_json_sorted(self, sorted, path):
        save_dict = dict(enumerate(sorted))
        with open(path, 'w') as json_file:
            json.dump(save_dict, json_file)

    def load_dos(self, dos):
        return glob(dos+"*.json")

    def load_dos_sorted(self, dos, sort_component=-1):
        json_dos_name = dos[:-1]
        json_path = f'{json_dos_name}{self.format}'
        if os.path.isfile(json_dos_name):
            sorted_path_json = self.load_json(json_path)

            if len(sorted_path_json) == len(os.listdir(dos))//2:
                sorted_paths = [sorted_path_json[i] for i in sorted_path_json]
                return sorted_paths

        new_sorted_paths = self.sort_paths_by_component(
            self.load_dos(dos), sort_component)
        self.save_json_sorted(new_sorted_paths, json_path)
        return new_sorted_paths

    def load_dataset(self, doss):
        doss_paths = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos(dos+"\\")
                doss_paths.append(paths)

        return doss_paths

    def generator_load_dataset(self, doss):
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                yield self.load_dos(dos+"\\")

    def load_dataset_sorted(self, doss):
        doss_paths = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                doss_paths.append(paths)
        return doss_paths

    def generator_load_dataset_sorted(self, doss):
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                yield self.load_dos_sorted(dos+"\\")

    def load_dataset_sequence(self, doss, max_interval=0.2):
        doss_sequences = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                paths_sequence = self.split_sorted_paths(
                    paths, time_interval=max_interval)
                doss_sequences.append(list(paths_sequence))

        return doss_sequences

    def generator_load_dataset_sequence(self, doss, max_interval=0.2):
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                yield self.generator_split_sorted_paths(
                    paths, time_interval=max_interval)

    def imgstring2json(self, dataset_obj, dst_dos, path):
        img = dataset_obj.load_image(path)
        annotations = dataset_obj.load_annotation(path)
        self.save_img_and_annotations(img, annotations, dos=dst_dos)

    def imgstring2dict2json(self, dataset_obj, dst_dos, path):
        img = dataset_obj.load_image(path)

        keys = [component.name for component in dataset_obj.label_structure]
        values = dataset_obj.load_annotation(path)
        annotations_dict = dict(zip(keys, values))
        self.save_img_and_annotations(img, annotations_dict, dos=dst_dos)

    def imgstring2json_encoded_deprecated(self, dataset_obj, dst_dos, path):
        annotations = dataset_obj.load_annotation(path)
        self.save_img_encoded_json_deprecated(dst_dos, path, annotations)

    def imgstring2json_dos(self, dataset_obj, imgstring2dos_function, src_dos, dst_dos):
        from tqdm import tqdm
        try:  # ugly way to create a dir if not exist
            os.mkdir(dst_dos)
        except OSError:
            pass

        paths = dataset_obj.load_dos(src_dos)
        for path in tqdm(paths):
            imgstring2dos_function(dataset_obj, dst_dos, path)

    def names2components(self, names):
        every_component = [direction_component,
                           speed_component, throttle_component, time_component]
        names_component_mapping = {}
        for component in every_component:
            names_component_mapping[component().name] = component

        label_structure = []
        for component_name in names:
            component = names_component_mapping.get(component_name)
            if component is not None:
                label_structure.append(component())
            else:
                raise ValueError('please enter a valid component name')

        return label_structure

    def indexes2components_names(self, indexes):
        return [self.get_component(i).name for i in indexes]


class direction_component:
    def __init__(self):
        self.name = "direction"
        self.type = float
        self.flip_factor = -1
        self.scale = 1.0
        self.offset = 0.0
        self.weight_acc = 0.1

    def get_item(self, json_data):
        return (self.type(json_data.get(self.name, 0.0))+self.offset)*self.scale

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
        self.flip_factor = 1
        self.scale = 1.0
        self.offset = 0.0
        self.weight_acc = 0.1

    def get_item(self, json_data):
        return (self.type(json_data.get(self.name, 0.0))+self.offset)*self.scale

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
        self.flip_factor = 1
        self.scale = 1.0
        self.offset = 0.0
        self.weight_acc = 0.1

    def get_item(self, json_data):
        return (self.type(json_data.get(self.name, 0.0))+self.offset)*self.scale

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
        self.flip_factor = 1

    def get_item(self, json_data):
        return self.type(json_data.get(self.name, time.time()))

    def add_item_to_dict(self, item, annotations_dict):
        annotations_dict[self.name] = self.type(item)
        return annotations_dict


class img_path_component:
    def __init__(self):
        self.name = "img_path"
        self.type = str
        self.flip_factor = 1

    def get_item(self, json_data):
        return self.type(json_data[self.name])

    def add_item_to_dict(self, annotations_dict):
        time_cmp = annotations_dict.get('time', time.time())
        dos = annotations_dict.get('dos', "")
        img_path = os.path.normpath(f'{dos}/{time_cmp}.png')

        annotations_dict[self.name] = img_path
        return annotations_dict


class imgbase64_component:
    def __init__(self):
        self.name = "img_base64"
        self.type = str
        self.flip_factor = 1

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
