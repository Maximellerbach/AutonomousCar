import cv2
import numpy as np

import interface
import image_processing

cap = cv2.VideoCapture('F:\\video-fh4\\FtcBrYpjnA_Trim.mp4')
orb = cv2.ORB_create(nlevels=8, edgeThreshold=0)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

s = interface.screen(np.zeros((120, 160, 3)))
s_orb = interface.screen(np.zeros((120, 160, 3)))
s_m = interface.screen(np.zeros((120, 160, 3)))

ui = interface.ui(screens=[s, s_orb, s_m], name="img", dt=1)

first_img = cap.read()[1]
# first_img = cv2.resize(first_img, (960,540))

diff = np.zeros(first_img.shape[:-1])
orbimg = np.zeros(first_img.shape[:-1])

diffs = []
prevkps = []
prevdes = []

while(1):
    _, img = cap.read()
    cv2.imshow("img", img)
    # img = cv2.resize(img, (960,540))

    if len(diffs)>5:
        del diffs[0]
    diffs.append(diff)
    w = np.array([(1/i)**2 for i in range(len(diffs)+1, 1, -1)])
    diff = np.average(diffs, axis=0, weights=w)

    diff, gray = image_processing.get_diff(img, diff)
    kps, des = image_processing.get_keypoints(orb, diff, gray)

    if prevkps!=[]:
        matches = bf.match(np.array(des), np.array(prevdes))
        # matches = sorted(matches, key = lambda x:x.distance)

        # g2 = cv2.drawMatches(gray, kps, diff, prevkps, matches, gray)
        g2 = np.copy(gray)
        g2 = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)
        for match in matches:
            q, t= (match.queryIdx, match.trainIdx)
            d = image_processing.get_keypoints_distance(kps[q], prevkps[t])

            if d<10:
                x1, y1 = kps[q].pt
                x2, y2 = prevkps[t].pt

                cv2.line(g2, (int(x1),int(y1)), (int(x2),int(y2)), [255, 0, 0], thickness=2)

        cv2.imshow("g2", g2/255)
        cv2.waitKey(1)


    prevkps = np.array(kps)
    prevdes = np.array(des)

    # matches

    diff = gray
