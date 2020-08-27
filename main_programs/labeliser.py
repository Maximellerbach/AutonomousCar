import cv2
import time
import PySimpleGUI as sg


from custom_modules.datasets import dataset_json


class Labeliser():
    def __init__(self, Dataset, output_components, dos, mode="union"):
        self.Dataset = Dataset
        self.output_components = output_components
        self.output_components_names = self.Dataset.indexes2components_names(
            self.output_components)

        layout = [[sg.Text(f'working dir: {dos}')],
                  [sg.Image(filename="", key='__IMAGE__')]]
        layout += [[sg.Text(name), sg.InputText(key=name)]
                   for name in self.output_components_names]
        layout += [[sg.Button('save', key='__SAVE__'),
                    sg.Button('previous', key='__PREV__'),
                    sg.Button('next', key='__NEXT__'),
                    sg.Text('image {i}/{tot}'+' '*8, key='__INDEX__')]]  # add some blank spaces to avoid overflow

        # Create the Window and initialize it
        self.window = sg.Window('Labeliser UI', layout)
        self.window.read(timeout=0)
        self.main(dos, mode)

    def main(self, dos, mode):
        img_paths = self.Dataset.load_dos(dos, search_format='.png')
        annotation_paths = self.Dataset.load_dos_sorted(dos)
        annotation_img_paths = [self.Dataset.load_meta(annotation_path, to_list=False).get(
            'img_path') for annotation_path in annotation_paths]

        if mode == "difference":
            to_label_paths = set(annotation_img_paths) - set(img_paths)
            print(f'found {len(to_label_paths)} images not labelled')

        elif mode == "union":
            to_label_paths = set(annotation_img_paths) | set(img_paths)
            print(f'found {len(to_label_paths)} images in total')

        elif mode == "intersection":
            to_label_paths = set(annotation_img_paths) & set(img_paths)
            print(f'found {len(to_label_paths)} images labelled')

        i = 0
        to_label_paths = list(to_label_paths)
        to_label_len = len(to_label_paths)
        while(i < to_label_len):
            img_path = to_label_paths[i]
            img = cv2.imread(img_path)
            imgbytes = cv2.imencode(".png", img)[1].tobytes()
            self.window["__IMAGE__"].update(data=imgbytes)
            self.window["__INDEX__"].update(f'image {i+1}/{to_label_len}')

            try:
                annotation = self.Dataset.load_annotation_json_from_img(
                    img_path, to_list=False)
            except OSError:
                annotation = self.Dataset.load_annotation_img_string(
                    img_path, cmp_structure=['direction', 'time'])

            for component in self.output_components_names:
                self.window[component].update(annotation[component])

            while(True):
                event, values = self.window.read(timeout=500)
                if event == sg.WIN_CLOSED:
                    break

                elif event == "__NEXT__":
                    i += 1
                    i = i % to_label_len
                    break
                elif event == "__PREV__":
                    i = i-1 if i > 0 else to_label_len-1
                    break

                elif event == "__SAVE__":
                    to_save = annotation
                    for labelled_cmp in values:
                        to_save[labelled_cmp] = float(values[labelled_cmp])
                    to_save['dos'] = dos
                    to_save['img_path'] = img_path
                    self.Dataset.save_annotation_dict(to_save)
                    i += 1
                    break

        self.window.close()


if __name__ == "__main__":
    Dataset = dataset_json.Dataset(['direction', 'speed', 'throttle', 'time'])
    direction_comp = Dataset.get_component('direction')
    direction_comp.offset = -7
    direction_comp.scale = 1/4

    output_components = [0, 2]  # indexes to labelise

    labeliser = Labeliser(Dataset, output_components,
                          "C:\\Users\\maxim\\random_data\\1 ironcar driving\\", mode="union")
