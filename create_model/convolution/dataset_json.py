from glob import glob
import json

import cv2

class DatasetJson():
    def __init__(self, lab_structure):
        self.label_structure = [i() for i in lab_structure]

    def save_img_and_json(self, dos, img, annotations):
        path = dos+str(annotations[-1])
        annotations.insert(-1, path+'.png') # add the img_name component
        cv2.imwrite(annotations[-2], img)

        annotations_dict = self.annotations_to_dict(annotations)
        with open(path+'.json', 'w') as json_file:
            json.dump(annotations_dict, json_file)

    def annotations_to_dict(self, annotations):
        annotations_dict = {}
        for it, component in enumerate(self.label_structure):
            annotations_dict[component.name] = annotations[it]

        return annotations_dict

    def load_json(self, path):
        with open(path, "r") as json_file:
            json_data = json.load(json_file) # json -> dict
        return json_data
    
    def load_component_item(self, path, n_component):
        json_data = self.load_json(path)
        return self.label_structure[n_component].get_item(json_data)

    def load_annotations(self, path):
        json_data = self.load_json(path)
        annotations = []
        for component in self.label_structure:
            # key = component.name
            # lab_item = json_data[key]
            lab_item = component.get_item(json_data)
            annotations.append(lab_item)

        return annotations
    
    def load_img_ang_json(self, path):
        annotations = self.load_annotations(path)
        img = cv2.imread(path.replace('.json', '.png'))

        return img, annotations

    def sort_paths_by_component(self, paths, n_component):
        def sort_function(path):
            return self.load_component_item(path, n_component)

        paths = sorted(paths, key=sort_function)
        return paths

    def load_dos(self, dos):
        return glob(dos+"*.json")    

    def load_dos_sorted(self, dos):
        return self.sort_paths_by_component(self.load_dos(dos), -1)

    def load_dataset(self, doss):
        doss_paths = []
        for dos in glob(doss+"*"):
            paths = self.load_dos(dos+"\\")
            doss_paths.append(paths)

        return doss_paths

    def load_dataset_sorted(self, doss):
        doss_paths = []
        for dos in glob(doss+"*"):
            paths = self.load_dos_sorted(dos+"\\")
            doss_paths.append(paths)

        return doss_paths


class direction_component:
    def __init__(self):
        self.name = "direction"
        self.do_flip = True

    def get_item(self, json_data):
        return float(json_data[self.name])

class speed_component:
    def __init__(self):
        self.name = "speed"
        self.do_flip = False

    def get_item(self, json_data):
        return float(json_data[self.name])

class throttle_component:
    def __init__(self):
        self.name = "throttle"
        self.do_flip = False

    def get_item(self, json_data):
        return float(json_data[self.name])

class img_name_component:
    def __init__(self):
        self.name = "img_name"
        self.do_flip = False

    def get_item(self, json_data):
        return str(json_data[self.name])

class time_component:
    def __init__(self):
        self.name = "time"
        self.do_flip = False

    def get_item(self, json_data):
        return float(json_data[self.name])
