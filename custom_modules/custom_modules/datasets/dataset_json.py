import ast
import base64
import io
import json
import os
import time
import typing
from enum import Enum
from glob import glob

import cv2
import numpy as np
from PIL import Image

from ..vis import vis_lab


def print_ret(func):
    def wrapped_f(*args, **kwargs):
        ret = func(*args, **kwargs)
        print(ret)
        return ret
    return wrapped_f


class Dataset():
    """Dataset class that contains everything needed to load and save a json dataset."""

    def __init__(self, lab_structure):
        """Load the given label structure.

        Args:
            lab_structure (list): list of components (class)
        """
        self.set_label_structure(lab_structure)
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
            self.__label_structure = [
                self.name2component(cmpt) for cmpt in lab_structure]
        else:
            self.__label_structure = [i() for i in lab_structure]

    def add_components(self, components_object):
        if isinstance(components_object, list):
            for component_object in components_object:
                self.add_components(component_object)
        else:
            if components_object.name not in self.get_label_structure_name():
                self.__label_structure.append(components_object)
                return

    def save_annotation_dict(self, annotation):
        """Save the annotation dict to {dos}{time}{self.format}.

        Args:
            annotation (dict): dict containing annotation
        """
        if len(annotation.keys()) != len(self.__label_structure):
            to_save_dict = {}
            for component in self.__label_structure:
                to_save_dict[component.name] = component.get_item(
                    annotation)
        else:
            to_save_dict = annotation

        dos = annotation.get('dos')
        assert dos is not None

        time_cmp = annotation.get('time')
        assert time_cmp is not None

        img_path = annotation.get('img_path')
        assert img_path is not None

        filename = f'{dos}{time_cmp}{self.format}'
        with open(os.path.normpath(filename), 'w') as json_file:
            json.dump(annotation, json_file)

        return filename

    def save_img_and_annotation(self, img, annotation, dos=None):
        """Save an image with it's annotation.

        Args:
            img (np.ndarray): image to be saved
            annotation (dict): dict containing annotation to be saved
            dos (string, optional): dos component if not set previously. Defaults to None.
        """
        if isinstance(annotation, list):
            if dos is None:
                raise "dos keyword should be specified"
            assert(len(annotation) == len(self.__label_structure))

            to_save_dict = {'dos': dos}
            for component, annotation in zip(self.__label_structure, annotation):
                to_save_dict = component.add_item_to_dict(
                    annotation, to_save_dict)

        elif isinstance(annotation, dict):
            # check if there is the expected number of labels
            if len(annotation.keys()) != len(self.__label_structure):
                to_save_dict = {}
                for component in self.__label_structure:
                    to_save_dict[component.name] = component.get_item(
                        annotation)
            else:
                to_save_dict = annotation

            if 'dos' not in to_save_dict.keys() and dos is not None:
                to_save_dict["dos"] = dos
            else:
                raise ValueError('dos keyword should be specified')

        else:
            raise ValueError(
                f'annotation should be a list or a dict, not {type(annotation)}')

        for component in self.__meta_components:
            to_save_dict = component.add_item_to_dict(to_save_dict)

        time_cmp = to_save_dict.get('time', time.time())
        cv2.imwrite(to_save_dict.get('img_path', ""), img)
        with open(os.path.normpath(f'{dos}{time_cmp}{self.format}'), 'w') as json_file:
            json.dump(to_save_dict, json_file)

    def save_img_and_annotation_deprecated(self, dos, img, annotation):
        path = dos+str(annotation[-1])  # save json with time component
        annotation.insert(-1, path+'.png')  # add the img_path component
        cv2.imwrite(annotation[-2], img)

        annotation_dict = self.annotation_to_dict_deprecated(annotation)
        with open(path+self.format, 'w') as json_file:
            json.dump(annotation_dict, json_file)

    def save_img_encoded_json_deprecated(self, dos, imgpath, annotation):
        path = dos+str(annotation[-1])
        component = self.get_component('img_base64')
        annotation.insert(-1, component.encode_img(imgpath))

        annotation_dict = self.annotation_to_dict_deprecated(annotation)
        with open(path+self.format, 'w') as json_file:
            json.dump(annotation_dict, json_file)

    def annotation_to_dict_deprecated(self, annotation):
        annotation_dict = {}
        for it, component in enumerate(self.__label_structure):
            annotation_dict[component.name] = annotation[it]

        return annotation_dict

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
            return list([get_item_comp(comp) for comp in self.__label_structure])
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
        annotation = self.load_annotation(path, to_list=to_list)
        img_path = self.load_meta(path, to_list=False).get('img_path')
        if img_path is None:
            raise ValueError('json does not have "img_path" component')
        img = cv2.imread(img_path)
        return img, annotation

    def load_img(self, path):
        img_path = self.load_meta(path, to_list=False).get('img_path')
        return cv2.imread(img_path)

    def load_annotation_json_from_img(self, img_path, to_list=True):
        annotation_path = img_path.split('.png')[0] + self.format
        return self.load_annotation(annotation_path, to_list=to_list)

    def make_to_pred(self, paths, input_components):
        xbatch = []
        ybatch = []
        for _ in range(len(self.get_label_structure_name())):
            ybatch.append([])

        for path in paths:
            img, annotation = self.load_img_and_annotation(path)
            xbatch.append(img)
            for it, lab in enumerate(annotation):
                ybatch[it].append(lab)

        to_pred = [np.array(xbatch, dtype=np.float32)/255]

        for i in input_components:
            to_pred.append(np.float32([np.float32(tmp_array)
                                       for tmp_array in ybatch[i]]))
        return to_pred

    def load_annotation_img_string(self, img_path, cmp_structure=['direction', 'time']):
        split_img_path = img_path.split('\\')[-1].split('.png')[0]
        components_value = split_img_path.split('_')
        tmp_annotation = {}
        for cmp_name, value in zip(cmp_structure, components_value):
            tmp_annotation[cmp_name] = float(value)

        annotation = {}
        for component in self.__label_structure:
            annotation[component.name] = component.get_item(tmp_annotation)

        return annotation

    def get_annotation_template(self):
        dummy_dict = {}
        template_dict = {}
        for component in self.__label_structure:
            template_dict[component.name] = component.get_item(dummy_dict)
        return template_dict

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

    def load_dos(self, dos, search_format='default'):
        if search_format == 'default':
            return glob(f"{dos}*{self.format}")
        else:
            return glob(f"{dos}*{search_format}")

    def load_dos_sorted(self, dos, sort_component=-1):
        json_dos_name = dos[:-1]
        json_path = f'{json_dos_name}{self.format}'
        if os.path.isfile(json_path):
            sorted_path_json = self.load_json(json_path)

            if len(sorted_path_json) == len(os.listdir(dos))//2:
                sorted_paths = [sorted_path_json[i] for i in sorted_path_json]
                return sorted_paths

        new_sorted_paths = self.sort_paths_by_component(
            self.load_dos(dos), sort_component)
        self.save_json_sorted(new_sorted_paths, json_path)
        return new_sorted_paths

    def load_doss(self, base_dos: str, doss_name: str, search_format='default'):
        doss = [base_dos+dos_name+"\\" for dos_name in doss_name]
        return [self.load_dos(dos, search_format=search_format) for dos in doss]

    def load_doss_sorted(self, base_dos: str, doss_name: str, sort_component=-1):
        doss = [base_dos+dos_name+"\\" for dos_name in doss_name]
        return [self.load_dos_sorted(dos) for dos in doss]

    def load_dataset(self, doss: str, flat=False):
        doss_paths = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos(dos+"\\")
                if flat:
                    doss_paths += paths
                else:
                    doss_paths.append(paths)
                print('loaded dos', dos)

        return doss_paths

    def generator_load_dataset(self, doss: str):
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                yield self.load_dos(dos+"\\")

    def load_dataset_sorted(self, doss: str):
        doss_paths = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                doss_paths.append(paths)
                print('loaded dos', dos)

        return doss_paths

    def generator_load_dataset_sorted(self, doss: str):
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                yield self.load_dos_sorted(dos+"\\")

    def load_dataset_sequence(self, doss: str, max_interval=0.2):
        doss_sequences = []
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                paths_sequence = self.split_sorted_paths(
                    paths, time_interval=max_interval)
                doss_sequences.append(list(paths_sequence))
                print('loaded dos', dos)

        return doss_sequences

    def generator_load_dataset_sequence(self, doss: str, max_interval=0.2):
        for dos in glob(doss+"*"):
            if os.path.isdir(dos):
                paths = self.load_dos_sorted(dos+"\\")
                yield self.generator_split_sorted_paths(
                    paths, time_interval=max_interval)

    def imgstring2json(self, dataset_obj, dst_dos, path):
        img = dataset_obj.load_image(path)
        annotation = dataset_obj.load_annotation(path)
        self.save_img_and_annotation(img, annotation, dos=dst_dos)

    def imgstring2dict2json(self, dataset_obj, dst_dos, path):
        img = dataset_obj.load_image(path)

        keys = [component.name for component in dataset_obj.label_structure]
        values = dataset_obj.load_annotation(path)
        annotation_dict = dict(zip(keys, values))
        self.save_img_and_annotation(img, annotation_dict, dos=dst_dos)

    def imgstring2json_encoded_deprecated(self, dataset_obj, dst_dos, path):
        annotation = dataset_obj.load_annotation(path)
        self.save_img_encoded_json_deprecated(dst_dos, path, annotation)

    def imgstring2json_dos(self, dataset_obj, imgstring2dos_function, src_dos, dst_dos):
        from tqdm import tqdm
        try:  # ugly way to create a dir if not exist
            os.mkdir(dst_dos)
        except OSError:
            pass

        paths = dataset_obj.load_dos(src_dos)
        for path in tqdm(paths):
            imgstring2dos_function(dataset_obj, dst_dos, path)

    def get_label_structure_name(self):
        return [i.name for i in self.__label_structure]

    def get_iterable_components(self):
        return [i for i in self.__label_structure if i.iterable]

    def indexes2components_names(self, indexes: typing.List[int]):
        return [self.get_component(i).name for i in indexes]

    def name2component(self, component_name: str):
        mapping = self.names_component_mapping()
        component = mapping.get(component_name)
        if component is not None:
            return component()
        raise ValueError('please enter a valid component name')

    def components_names2indexes(self):
        mapping = {}
        for it, name in enumerate(self.get_label_structure_name()):
            mapping[name] = it
        return mapping

    def names_component_mapping(self):
        mapping = {}
        for component in every_component:
            mapping[component.name] = component.value
        return mapping


class direction_component:
    def __init__(self):
        self.name = "direction"
        self.type = float
        self.default = 0.0

        self.flip = True
        self.scale = 1.0
        self.offset = 0.0
        self.weight_acc = 0.1
        self.iterable = False
        self.is_couple = False
        self.vis_func = vis_lab.direction

    def get_item(self, json_data):
        return (self.type(json_data.get(self.name, self.default))+self.offset)*self.scale

    def add_item_to_dict(self, item, annotation_dict: dict):
        if isinstance(item, self.type):
            annotation_dict[self.name] = item
        else:
            ValueError(f'item type: {type(item)} should match {self.type}')
        return annotation_dict

    def from_string(self, string):
        return self.type(string)

    def flip_item(self, item):
        return item*-1


class speed_component:
    def __init__(self):
        self.name = "speed"
        self.type = float
        self.default = 1.0
        self.iterable = False

        self.flip = False
        self.scale = 1.0
        self.offset = 0.0
        self.weight_acc = 0.1
        self.is_couple = False

    def get_item(self, json_data):
        return (self.type(json_data.get(self.name, self.default))+self.offset)*self.scale

    def add_item_to_dict(self, item, annotation_dict: dict):
        if isinstance(item, self.type):
            annotation_dict[self.name] = item
        else:
            ValueError(f'item type: {type(item)} should match {self.type}')
        return annotation_dict

    def from_string(self, string):
        return self.type(string)


class throttle_component:
    def __init__(self):
        self.name = "throttle"
        self.type = float
        self.default = 0.5
        self.iterable = False

        self.flip = False
        self.scale = 1.0
        self.offset = 0.0
        self.weight_acc = 0.1
        self.is_couple = False
        self.vis_func = vis_lab.throttle

    def get_item(self, json_data: dict):
        return (self.type(json_data.get(self.name, self.default))+self.offset)*self.scale

    def add_item_to_dict(self, item, annotation_dict: dict):
        if isinstance(item, self.type):
            annotation_dict[self.name] = item
        else:
            ValueError(f'item type: {type(item)} should match {self.type}')
        return annotation_dict

    def from_string(self, string):
        return self.type(string)


class right_lane_component:
    def __init__(self):
        self.name = "right_lane"
        self.type = np.float32
        self.xnorm = 80
        self.ynorm = 60
        self.normarray = np.array(
            [self.xnorm, self.ynorm, self.xnorm, self.ynorm])
        self.fliparray = np.array(
            [1, 0, 1, 0])
        self.default = np.array([[0, 0], [0, 0]], dtype=self.type)
        self.default_flat = np.array([0, 0, 0, 0], dtype=self.type)

        self.flip = True
        self.iterable = True
        self.weight_acc = 5
        self.couple = 'left_lane'
        self.is_couple = True
        self.vis_func = vis_lab.lane

    def get_item(self, json_data: dict):
        pts_list = json_data.get(self.name)
        if pts_list is not None:
            return (np.array(pts_list[0]+pts_list[1], dtype=self.type) / self.normarray) - 1
        else:
            return self.default

    def add_item_to_dict(self, item, annotation_dict: dict):
        annotation_dict[self.name] = self.type(item)
        return annotation_dict

    def from_string(self, string):
        return ast.literal_eval(string)

    def flip_item(self, item):  # considering that item is normalised
        return item*-self.fliparray


class left_lane_component:
    def __init__(self):
        self.name = "left_lane"
        self.type = np.float32
        self.default = [[0, 0], [0, 0]]
        self.default_flat = [0, 0, 0, 0]
        self.xnorm = 80
        self.ynorm = 60
        self.normarray = np.array(
            [self.xnorm, self.ynorm, self.xnorm, self.ynorm])
        self.fliparray = np.array(
            [1, 0, 1, 0])

        self.flip = True
        self.iterable = True
        self.weight_acc = 5
        self.couple = 'right_lane'
        self.is_couple = True
        self.vis_func = vis_lab.lane

    def get_item(self, json_data: dict):
        pts_list = json_data.get(self.name)
        if pts_list is not None:
            return (np.array(pts_list[0]+pts_list[1], dtype=self.type) / self.normarray) - 1
        else:
            return self.default

    def add_item_to_dict(self, item, annotation_dict: dict):
        annotation_dict[self.name] = self.type(item)
        return annotation_dict

    def from_string(self, string):
        return ast.literal_eval(string)

    def flip_item(self, item):  # considering that item is normalised
        return item*-self.fliparray


class img_path_component:
    def __init__(self):
        self.name = "img_path"
        self.type = str
        self.flip = False
        self.is_couple = False

    def get_item(self, json_data: dict):
        return self.type(json_data[self.name])

    def add_item_to_dict(self, annotation_dict: dict):
        time_cmp = annotation_dict.get('time', time.time())
        dos = annotation_dict.get('dos', "")
        img_path = os.path.normpath(f'{dos}/{time_cmp}.png')

        annotation_dict[self.name] = img_path
        return annotation_dict


class time_component:
    def __init__(self):
        self.name = "time"
        self.type = float
        self.default = time.time
        self.iterable = False

        self.flip = False
        self.is_couple = False

    def get_item(self, json_data: dict):
        return self.type(json_data.get(self.name, self.default()))

    def add_item_to_dict(self, item: float, annotation_dict: dict):
        annotation_dict[self.name] = self.type(item)
        return annotation_dict


class imgbase64_component:
    def __init__(self):
        self.name = "img_base64"
        self.type = str
        self.flip = False
        self.is_couple = False

    def get_item(self, json_data: dict):
        # getting the image from string and convert it to BGR
        base64_img = self.type(json_data[self.name])
        base64_img_decoded = base64.b64decode(base64_img)
        return cv2.cvtColor(np.array(Image.open(io.BytesIO(base64_img_decoded))), cv2.COLOR_RGB2BGR)

    def encode_img(self, imgpath: str):
        with open(imgpath, "rb") as img_file:
            img_encoded = base64.b64encode(img_file.read()).decode('utf-8')
        return img_encoded

    def add_item_to_dict(self, img_path: str, annotation_dict: dict):
        encoded = self.encode_img(img_path)
        annotation_dict[self.name] = encoded

        return annotation_dict


class every_component(Enum):  # not used for the moment
    direction = direction_component
    speed = speed_component
    throttle = throttle_component
    time = time_component
    right_lane = right_lane_component
    left_lane = left_lane_component
