import dataset
import dataset_json
import os

def imgstring2json(dataset_obj, dataset_json_obj, dst_dos, path):
    img = dataset_obj.load_image(path)
    annotations = dataset_obj.load_annotation(path)
    dataset_json_obj.save_img_and_json(dst_dos, img, annotations)

def imgencoded2json(dataset_obj, dataset_json_obj, dst_dos, path):
    annotations = dataset_obj.load_annotation(path)
    dataset_json_obj.save_img_encoded_json(dst_dos, path, annotations)

def imgstring2json_dos(dataset_obj, dataset_json_obj, src_dos, dst_dos):
    try:
        os.mkdir(dst_dos)
    except:
        pass
    paths = dataset_obj.load_dos(src_dos)
    for path in tqdm(paths):
        imgstring2json(Dataset, DatasetJson, dst_dos, path)

if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob

    Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])
    DatasetJson = dataset_json.DatasetJson([dataset_json.direction_component, dataset_json.img_name_component, dataset_json.time_component])

    src_dos = "C:\\Users\\maxim\\datasets\\"
    dst_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"
    doss = glob(src_dos+"*")

    # for dos in doss:
    #     dos_name = dos.split('\\')[-1]
    #     imgstring2json_dos(Dataset, DatasetJson, dos+"\\", dst_dos+dos_name+"\\")

    for dos in doss:
        sorted_paths = DatasetJson.load_dos_sorted(dos+"\\")
