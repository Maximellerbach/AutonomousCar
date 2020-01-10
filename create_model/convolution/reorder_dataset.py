import autolib
import os
from glob import glob

import numpy as np
from tqdm import tqdm
import cv2

dire = [3, 5, 7, 9, 11]

def pack_datasets(dos, new_dos, d_threshold=60):
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
    name = p.split('\\')[-1]
    lab, date = name.split('_')
    date = date.split('.png')[0]
    return date

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
        date = get_date(p)
        dates.append([str2float(date, p, dos), p])

    dates.sort(key= lambda x: x[0])
    return dates

def load_dataset(dos):
    folds = glob(dos+"*")
    dataset = []
    datalen = 0
    for folder in folds:
        g = [[p, get_date(p)] for p in glob(folder+'\\*')]
        g.sort(key= lambda x:x[1])
        d = [i[0] for i in g]
        dataset.append(d)
        datalen+=len(d)

    return np.array(dataset), datalen


if __name__ == "__main__":
    pack_datasets('C:\\Users\\maxim\\image_mix2\\', 'C:\\Users\\maxim\\datasets\\', 1000)
    # dts, datalen = load_dataset('C:\\Users\\maxim\\datasets\\')
