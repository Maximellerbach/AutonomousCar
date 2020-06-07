import cv2
from glob import glob

class Dataset():
    def __init__(self, lab_structure):
        self.label_structure = [i(it) for it, i in enumerate(lab_structure)]

    def split_string(self, img_string, sep1="\\", sep2="_", img_format=".png"):
        return img_string.split("\\")[-1].split(img_format)[0].split("_")

    def load_component_item(self, img_string, n_component):
        split_string = self.split_string(img_string)
        return self.label_structure[n_component].get_item(split_string)

    def load_image(self, img_string):
        return cv2.imread(img_string)

    def load_annotation(self, img_string):
        split_string = self.split_string(img_string)
        annotations = [label_type.get_item(split_string) for label_type in self.label_structure]
        return annotations

    def repeat_function(self, function, items):
        return [function(item) for item in items]

    def sort_paths_by_component(self, paths, n_component):
        def sort_function(string):
            return self.label_structure[n_component].get_item(self.split_string(string))

        paths = sorted(paths, key=sort_function)
        return paths

    def load_dataset_sorted(self, doss, sort_component=-1):
        sorted_doss = []
        for dos in glob(doss+"*"):
            paths = glob(dos+"\\*")
            sorted_doss.append(self.sort_paths_by_component(paths, sort_component)) # -1 is time component

        return sorted_doss
    
    def load_dos_sorted(self, dos, sort_component=-1):
        paths = glob(dos+"*")
        return self.sort_paths_by_component(paths, sort_component) # -1 is time component
    
    def load_dos(self, dos):
        return glob(dos+"*")
    
    def load_dataset(self, doss):
        doss_paths = []
        for dos in glob(doss+"*"):
            paths = glob(dos+"\\*")
            doss_paths.append(paths) # -1 is time component

        return doss_paths

class direction_component:
    def __init__(self, n_component):
        self.name = "direction"
        self.index_in_string = n_component
        self.do_flip = True

    def get_item(self, split_string):
        item = split_string[self.index_in_string]
        return float(item)

class speed_component:
    def __init__(self, n_component):
        self.name = "speed"
        self.index_in_string = n_component
        self.do_flip = False

    def get_item(self, split_string):
        item = split_string[self.index_in_string]
        return float(item)

class throttle_component:
    def __init__(self, n_component):
        self.name = "throttle"
        self.index_in_string = n_component
        self.do_flip = False

    def get_item(self, split_string):
        item = split_string[self.index_in_string]
        return float(item)

class time_component:
    def __init__(self, n_component):
        self.name = "time"
        self.index_in_string = n_component
        self.do_flip = False

    def get_item(self, split_string):
        item = split_string[self.index_in_string]
        return float(item)



def angle_speed_to_throttle(dos, target_speed=18):
    dataset = Dataset([direction_component, speed_component, time_component])

    for path in glob(dos+"*"):
        annotations = dataset.load_annotation(path)
        print(annotations)

    return 


if __name__ == "__main__":
    angle_speed_to_throttle("")
