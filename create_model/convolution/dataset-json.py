from glob import glob
import json

import cv2


class DatasetJson():
    def __init__(self, lab_structure):
        self.label_structure = [i(it) for it, i in enumerate(lab_structure)]

    def save_img_and_json(self, dos, img, annotations):
        path = dos+annotations[-1]
        cv2.imwrite(path+'.png', img)

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
        return json_data[self.label_structure[n_component].name]

    def load_annotations(self, path):
        json_data = self.load_json(path)
        annotations = []
        for component in self.label_structure:
            key = component.name
            lab_item = json_data[key]
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
    def __init__(self, n_component):
        self.name = "direction"
        self.index_in_json = n_component
        self.do_flip = True

class speed_component:
    def __init__(self, n_component):
        self.name = "speed"
        self.index_in_json = n_component
        self.do_flip = False

class throttle_component:
    def __init__(self, n_component):
        self.name = "throttle"
        self.index_in_json = n_component
        self.do_flip = False

class time_component:
    def __init__(self, n_component):
        self.name = "time"
        self.index_in_json = n_component
        self.do_flip = False
