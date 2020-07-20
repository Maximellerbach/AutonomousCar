import dataset
import dataset_json


def imgstring2json(dataset_obj, dataset_json_obj, dos, path):
    annotations = dataset_obj.load_annotation(path)
    img = dataset_obj.load_image(path)

    dataset_json_obj.save_img_and_json(dos, img, annotations)

if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob
    
    Dataset = dataset.Dataset([dataset.direction_component, dataset.time_component])
    DatasetJson = dataset_json.DatasetJson([dataset_json.direction_component, dataset_json.img_name_component, dataset_json.time_component])

    src_dos = "C:\\Users\\maxim\\datasets\\11 sim circuit 2\\"
    dst_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\"
    
    paths = Dataset.load_dos(src_dos)
    for path in tqdm(paths):
        imgstring2json(Dataset, DatasetJson, dst_dos, path)

    
