from custom_modules.datasets import dataset_json, dataset_sql

datasetJson = dataset_json.Dataset(["direction", "speed", "throttle", "time"])
datasetSQL = dataset_sql.Dataset()


src_dos = "C:\\Users\\maxim\\random_data\\json_dataset\\20 checkpoint patch"

dataset_name = src_dos.split('\\')[-1]
paths = datasetJson.load_dos_sorted(src_dos+"\\")
for path in paths:
    img, annotation = datasetJson.load_img_and_annotation(path, to_list=False)
    meta = datasetJson.load_meta(path, to_list=False)
    annotation['img_path'] = meta['img_path']
    datasetSQL.save_annotation_dict(annotation, dataset_name=dataset_name)

paths = datasetSQL.load_dataset_sorted("20 checkpoint patch")
print(paths, len(paths))
print(datasetSQL.rows_meta)
