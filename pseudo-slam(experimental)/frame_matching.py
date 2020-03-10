from glob import glob

import cv2
import numpy as np

import frame
import image_processing

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING2)

p1dos = 'C:\\Users\\maxim\\datasets\\2\\*' # 
p2dos = 'C:\\Users\\maxim\\datasets\\1\\*'

g1dos = glob(p1dos)
g2dos = glob(p2dos)
H, W = cv2.imread(g1dos[0]).shape[:2]
H *= 2
W *= 2

for i in range(0, len(g1dos), 1):
    img1 = cv2.resize(cv2.imread(g1dos[i]), (W, H))
    img1 = image_processing.add_padding(img1, n=(20,20))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # p2 = gdos[i+1]
    img2 = cv2.resize(cv2.imread(g2dos[i]), (W, H))
    img2 = image_processing.add_padding(img2, n=(20,20))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    f1 = frame.Frame()
    f2 = frame.Frame()

    f1.kps, f1.des = image_processing.extractFeatures(orb, gray1, gray1, name="1")
    f2.kps, f2.des = image_processing.extractFeatures(orb, gray2, gray2, name="2")
    
    if len(f2.kps)>2:
        f1.matches = bf.knnMatch(np.array(f1.des), np.array(f2.des), k=2)
        g2 = np.copy(img1)

        for match, n in f1.matches:
            q, t= (match.queryIdx, match.trainIdx)    
            d = image_processing.get_kps_distance(f1.kps[q], f2.kps[t])
            if match.distance < 0.7*n.distance:
                x1, y1, = f1.kps[q].pt

                x2, y2 = f2.kps[t].pt
                x2 = x2/img2.shape[1]
                y2 = y2/img2.shape[0]


                cv2.line(g2, (int(x1),int(y1)), (int(x2*img1.shape[1]),int(y2*img1.shape[0])), [255, 0, 0], thickness=2)

        cv2.imshow("g2", g2/255)
        cv2.waitKey(0)
    cv2.waitKey(1)