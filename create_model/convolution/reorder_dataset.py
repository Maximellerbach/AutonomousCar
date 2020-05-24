
import json
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

import autolib

dire = [3, 5, 7, 9, 11]

def pack_datasets(dos, new_dos, d_threshold=60):
    try:
        os.mkdir(new_dos)
    except:
        pass
    
    dates = sort_by_date(dos)

    gaps = []
    for it, i in enumerate(dates):
        if it == 0:
            prev = i[0]
        else:
            gap = np.absolute(i[0]-prev)
            if gap>d_threshold:
                gaps.append(it)
            prev = i[0]

    print(gaps) # auto-detected gaps : differents datasets

    datasets = [[]]
    d = 0
    for it, i in enumerate(dates):
        if it in gaps:
            d+=1
            datasets.append([])
        datasets[d].append(i)

    print(len(datasets))
    for dataset in datasets:
        print(len(dataset)) 

    for d, dataset in enumerate(datasets):
        try:
            os.mkdir(new_dos+str(d)+"\\")
        except:
            pass

        tmp_save = new_dos+str(d)+"\\"
        g = glob(dos+"*")
        ps = [str(get_date(p)) for p in g]

        for it, i in tqdm(enumerate(dataset)):
            lab = dire[autolib.get_label(i[1], flip=False)[0]]
            im = cv2.imread(i[1])
            cv2.imwrite(tmp_save+str(lab)+"_"+str(i[0])+'.png', im)
            
def get_date(p):
    try:
        name = p.split('\\')[-1]
        lab = name.split('_')[0]
        date = name.split('_')[-1]
        date = date.split('.png')[0]
        return float(date)
    except Exception as e:
        print(p, e)
        
def get_speed(p):
    name = p.split('\\')[-1]
    lab, speed, date = name.split('_')
    return float(speed)

def str2float(string, ancient_str, relative_path):
    try:
        f = float(string)
    except:
        i, r = string.split('.')
        r, occ = r.split(' ')
        occ = occ.split('(')[1].split(')')[0]
        reconstructed = i+'.'+r+occ
        f = float(reconstructed)
        lab = dire[autolib.get_label(ancient_str, flip=False)[0]]
        im = cv2.imread(ancient_str)
        cv2.imwrite(relative_path+str(lab)+"_"+str(reconstructed)+'.png', im)
        os.remove(ancient_str)
        print(ancient_str)

    return f

def sort_by_date(dos):
    dates = []
    for p in tqdm(glob(dos+'*')):
        date = str(get_date(p))
        dates.append([str2float(date, p, dos), p])

    dates.sort(key= lambda x: x[0])
    return dates

def load_dataset(dos, recursive=True):
    dataset = []
    datalen = 0
    if recursive:
        folds = glob(dos+"*")
        for folder in folds:
            g = glob(folder+'\\*')
            d = sorted(g, key = get_date)
            dataset.append(d)
            datalen+=len(d)
    else:
        g = glob(dos+'*')
        dataset = sorted(g, key = get_date)
        datalen = len(dataset)

    return np.array(dataset), datalen

def json2angles(dirpath):
    angles = []
    img_paths = glob(dirpath+'*.jpg')
    dir_p = dirpath.split('*')[0]
    for path in img_paths:
        number = path.split('\\')[-1].split('_')[0]
        json_path = dir_p+'record_'+number+'.json'
        with open(json_path) as j:
            data = json.load(j)
            angle = data["user/angle"]
        angles.append(angle)

    return angles

if __name__ == "__main__":
    angles = json2angles('C:\\Users\\maxim\\gen_track_user_drv_right_lane\\')
    print(angles)

