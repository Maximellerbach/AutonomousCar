from glob import glob

import cv2


class Dataset():
    """Deprecated dataset class."""

    def __init__(self, lab_structure):
        self.label_structure = [i(it) for it, i in enumerate(lab_structure)]
        self.format = '.png'

    def split_string(self, img_string, sep1="\\", sep2="_", img_format=".png"):
        return img_string.split("\\")[-1].split(img_format)[0].split("_")

    def load_component_item(self, img_string, n_component):
        split_string = self.split_string(img_string)
        return self.label_structure[n_component].get_item(split_string)

    def load_image(self, img_string):
        return cv2.imread(img_string)

    def load_annotation(self, img_string):
        def get_item_comp(component):
            return component.get_item(split_string)

        split_string = self.split_string(img_string)
        return list(map(get_item_comp, self.label_structure))

    def load_img_ang_annotation(self, path):
        annotation = self.load_annotation(path)
        img = self.load_image(path)

        return img, annotation

    def repeat_function(self, function, items):
        return [function(item) for item in items]

    def sort_paths_by_component(self, paths, n_component):
        def sort_function(string):
            return self.label_structure[n_component].get_item(self.split_string(string))

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

    def load_dataset_sorted(self, doss, sort_component=-1):
        sorted_doss = []
        for dos in glob(doss+f"*{self.format}"):
            paths = self.load_dos(dos+"\\")
            sorted_doss.append(self.sort_paths_by_component(
                paths, sort_component))  # -1 is time component

        return sorted_doss

    def load_dos_sorted(self, dos, sort_component=-1):
        paths = self.load_dos(dos)
        # -1 is time component
        return self.sort_paths_by_component(paths, sort_component)

    def load_dos(self, dos):
        return glob(dos+f"*{self.format}")

    def load_dataset(self, doss):
        doss_paths = []
        for dos in glob(doss+f"*{self.format}"):
            paths = self.load_dos(dos+"\\")
            doss_paths.append(paths)

        return doss_paths

    def load_dataset_sequence(self, doss, max_interval=0.2):
        doss_sequences = []
        for dos in glob(doss+f"*{self.format}"):
            paths = self.load_dos_sorted(dos+"\\")
            paths_sequence = self.split_sorted_paths(
                paths, time_interval=max_interval)
            doss_sequences.append(list(paths_sequence))

        return doss_sequences


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


def annotation_to_name(annotation):
    string = ""
    for it, component in enumerate(annotation):
        string += str(component)

        if it != len(annotation)-1:
            string += "_"
        else:
            string += ".png"

    return string


def save(save_path, dts, annotation):
    import os
    import time

    from tqdm import tqdm
    import shutil

    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)

    for i in tqdm(range(len(dts))):
        time.sleep(0.0001)
        name = annotation_to_name(annotation[i])
        shutil.copy(dts[i], save_path+name)


# to transform old data format into new ones
def angle_speed_to_throttle(dos, target_speed=18, max_throttle=1, min_throttle=0.45):
    # Function from my Virtual Racing repo
    def opt_acc(st, current_speed, max_throttle, min_throttle, target_speed):
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

    dts = glob(dos+"*.png")
    Y = []
    for path in dts:
        annotation = dataset.load_annotation(path)
        converted_throttle = opt_acc(
            annotation[0], annotation[1], max_throttle, min_throttle, target_speed)
        annotation.insert(2, converted_throttle)

        Y.append(annotation)
    return dts, Y


def add_dummy_speed(dos, dummy_speed=10):
    dataset = Dataset([direction_component, time_component])

    dts = glob(dos+"*.png")
    Y = []
    for path in dts:
        annotation = dataset.load_annotation(path)
        annotation.insert(1, dummy_speed)
        Y.append(annotation)

    return dts, Y


def cat2linear_dataset(dos):
    dataset = Dataset([direction_component, time_component])

    dts = glob(dos+"*.png")
    Y = []
    for path in dts:
        annotation = dataset.load_annotation(path)
        annotation[0] = cat2linear(annotation[0])
        Y.append(annotation)

    return dts, Y


def cat2linear(ny):
    return (ny-7)/4


if __name__ == "__main__":
    import os
    base_path = os.getenv('ONEDRIVE') + "\\random_data"

    dts, annotation = cat2linear_dataset(
        f"{base_path}\\11 sim circuit 2\\")
    save(f"{base_path}\\linear\\11 sim circuit 2\\", dts, annotation)
