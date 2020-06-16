from glob import glob

import cv2


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

def annotations_to_name(annotations):
    string = ""
    for it, component in enumerate(annotations):
        string+=str(component)

        if it != len(annotations)-1:
            string+="_"
        else:
            string+=".png"

    return string

def save(save_path, dts, annotations):
    import os
    import time

    from tqdm import tqdm
    import shutil

    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)

    for i in tqdm(range(len(dts))):
        time.sleep(0.0001)
        name = annotations_to_name(annotations[i])
        shutil.copy(dts[i], save_path+name)

def angle_speed_to_throttle(dos, target_speed=18, max_throttle=1, min_throttle=0.45):
    def opt_acc(st, current_speed, max_throttle, min_throttle, target_speed): # Function from my Virtual Racing repo
        dt_throttle = max_throttle-min_throttle

        optimal_acc = ((target_speed-current_speed)/target_speed)
        if optimal_acc < 0:
            optimal_acc = 0

        optimal_acc = min_throttle+((optimal_acc**0.1)*(1-abs(st)))*dt_throttle

        if optimal_acc > max_throttle:
            optimal_acc = max_throttle
        elif optimal_acc < min_throttle:
            optimal_acc = min_throttle

        return optimal_acc

    dataset = Dataset([direction_component, speed_component, time_component])

    dts = glob(dos+"*")
    Y = []
    for path in dts:
        annotations = dataset.load_annotation(path)
        converted_throttle = opt_acc(annotations[0], annotations[1], max_throttle, min_throttle, target_speed)
        annotations.insert(2, converted_throttle)

        Y.append(annotations)
    return dts, Y

def add_dummy_speed(dos, dummy_speed=10):

    dataset = Dataset([direction_component, time_component])

    dts = glob(dos+"*")
    Y = []
    for path in dts:
        annotations = dataset.load_annotation(path)
        annotations.insert(1, dummy_speed)
        Y.append(annotations)

    return dts, Y


if __name__ == "__main__":
    # dts, annotations = add_dummy_speed("C:\\Users\\maxim\\random_data\\linear\\1 ironcar driving\\", dummy_speed=1)
    # save("C:\\Users\\maxim\\random_data\\speed\\1 ironcar driving\\", dts, annotations)

    dts, annotations = angle_speed_to_throttle("C:\\Users\\maxim\\random_data\\17 custom maps\\")
    save("C:\\Users\\maxim\\random_data\\throttle\\17 custom maps\\", dts, annotations)
    