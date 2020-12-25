import cv2
import time
import PySimpleGUI as sg


from custom_modules.datasets import dataset_json


class Labeliser():
    def __init__(self, Dataset, output_components, dos, mode="union"):
        self.Dataset = Dataset
        self.output_components = output_components
        self.dos = dos
        self.mode = mode

        self.output_components_names = self.Dataset.indexes2components_names(
            self.output_components)
        self.iterable_components = self.Dataset.get_iterable_components()

        self.img_paths = self.Dataset.load_dos(self.dos, search_format='.png')
        self.annotation_paths = self.Dataset.load_dos_sorted(self.dos)
        self.annotation_img_paths = [self.Dataset.load_meta(annotation_path, to_list=False).get(
            'img_path') for annotation_path in self.annotation_paths]

        self.img_paths_mapping = {}
        for annotation_img_path, annotation_path in zip(self.annotation_img_paths, self.annotation_paths):
            self.img_paths_mapping[annotation_img_path] = annotation_path

        layout = [[sg.Text(f'working dir: {self.dos}')],
                  [sg.Graph((160, 120), (0, 120), (160, 0), enable_events=True, key='__GRAPH__')]]

        layout += [[sg.Text(name), sg.InputText(key=name)]
                   for name in self.output_components_names]

        layout += [[sg.Button('save', key='__SAVE__'),
                    sg.Button('previous', key='__PREV__'),
                    sg.Button('next', key='__NEXT__'),
                    sg.Text('image {i}/{tot}'+' '*8, key='__INDEX__')]]  # add some blank spaces to avoid overflow

        # Create the Window and initialize it
        self.window = sg.Window('Labeliser UI', layout)
        self.window.read(timeout=0)
        self.main()

    def main(self):
        if self.mode == "difference":
            to_label_paths = set(
                self.annotation_img_paths) - set(self.img_paths)

        elif self.mode == "union":
            to_label_paths = set(
                self.annotation_img_paths) | set(self.img_paths)

        elif self.mode == "intersection":
            to_label_paths = set(
                self.annotation_img_paths) & set(self.img_paths)

        print(f'found {len(to_label_paths)} images to label')

        i = 0
        to_label_paths = list(to_label_paths)
        to_label_len = len(to_label_paths)
        while(i < to_label_len-1):
            pt_list_iteration = 0
            img_path = to_label_paths[i]
            self.window["__GRAPH__"].draw_image(
                filename=img_path, location=(0, 0))
            self.window["__GRAPH__"].set_cursor('hand2')

            self.window["__INDEX__"].update(f'image {i+1}/{to_label_len}')

            try:
                annotation_path = self.img_paths_mapping[img_path]
                annotation = self.Dataset.load_annotation(
                    annotation_path, to_list=False)
            except KeyError:
                annotation = self.Dataset.load_annotation_img_string(
                    img_path, cmp_structure=['direction', 'time'])  # default dataset structure

            for component_name in self.output_components_names:
                self.window[component_name].update(
                    str(annotation[component_name]))

            while(True):
                event, values = self.window.read()
                if event == sg.WIN_CLOSED:
                    break

                elif event == "__NEXT__":
                    i += 1
                    i = i % to_label_len
                    break
                elif event == "__PREV__":
                    i = i-1 if i > 0 else to_label_len-1
                    break

                elif event == "__GRAPH__" and values["__GRAPH__"] != (None, None):
                    if len(self.iterable_components) == 0:
                        continue
                    pt_value = values["__GRAPH__"]
                    component = self.iterable_components[pt_list_iteration // 2]
                    list_of_point = component.from_string(
                        values[component.name])
                    list_of_point[pt_list_iteration % 2] = pt_value
                    self.window[component.name].update(str(list_of_point))

                    pt_list_iteration = (
                        pt_list_iteration+1) % (2 * len(self.iterable_components))

                elif event == "__SAVE__":
                    to_save = annotation
                    for cmp_key in values:
                        if "__" not in cmp_key:
                            to_save[cmp_key] = self.Dataset.get_component(
                                cmp_key).from_string(values[cmp_key])

                    to_save['dos'] = self.dos
                    to_save['img_path'] = img_path
                    new_annotation_path = self.Dataset.save_annotation_dict(
                        to_save)
                    self.img_paths_mapping[img_path] = new_annotation_path
                    i += 1
                    break

        self.window.close()


if __name__ == "__main__":
    import os
    base_path = os.path.expanduser("~") + "\\random_data"
    path = f"{base_path}\\1 ironcar driving\\"
    path = 'C:\\Users\\maxim\\recorded_imgs\\0_1600008448.0622997\\'

    Dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle'])

    output_components = [0, 1, 2]  # indexes to labelise

    labeliser = Labeliser(Dataset, output_components,
                          path, mode="union")
