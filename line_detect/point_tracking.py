import os

import cv2
import numpy as np
import open3d
import threading

from frame import *
import image_processing
import points3D as p3D
from skimage.measure import ransac

cloud_points = p3D.cloud_points()

cap = cv2.VideoCapture('F:\\video-fh4\\MFAwpTjlSY_Trim.mp4') # F:\\video-fh4\\FtcBrYpjnA_Trim.mp4

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

W, H = (960, 640)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING2)

first_img = cap.read()[1]
first_img = cv2.resize(first_img, (W,H))

Fx = 500
Fy = 500

K = np.array([[Fx,0,W//2],[0,Fy,H//2],[0,0,1]])
Kinv = np.linalg.inv(K)

frames = [Frame()]
eps_pose = np.eye(4)
eps_t = [0, 0, 0]

it = 0
while cap.isOpened():
    it += 1

    frames.append(Frame())
    f1 = frames[-1]
    f2 = frames[-2]

    img, gray = image_processing.capture_img(cap, (H, W))
    f1.kps, f1.des = image_processing.extractFeatures(orb, gray, gray)

    if len(frames)>2:
        f1.matches = bf.knnMatch(np.array(f1.des), np.array(f2.des), k=2)

        g2 = np.copy(gray)
        g2 = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)

        idx1 = []
        idx2 = []
        ret = []

        for match, n in f1.matches:
            if match.distance < 0.75*n.distance:
                q, t= (match.queryIdx, match.trainIdx)
                d = image_processing.get_kps_distance(f1.kps[q], f2.kps[t])

                if 0<d<50:
                    f1.idx1.append(q)
                    f1.idx2.append(t)
                    
                    x1, y1, = f1.kps[q].pt
                    x2, y2 = f2.kps[t].pt

                    p1 = (x1/W, y1/H, 0)
                    p2 = (x2/W, y2/H, 0)

                    f1.pts.append([p1[:2], p2[:2]])
                    p = p3D.point(f1.kps[q], t)
                    p.set_point(p1)
                    f1.points.append(p)

                    cv2.line(g2, (int(x1),int(y1)), (int(x2),int(y2)), [255, 0, 0], thickness=2)
        cv2.imshow("g2", g2/255)

        f1.to_array()
        p = image_processing.normalize(Kinv, get_points(f1), n=2)
        set_points(f1, p)
        get_matching(f1, f2, n=2, img_shape=(W, H))
        
        # normalized 2D points
        # f1.pts[:, 0] = image_processing.normalize(Kinv, f1.pts[:, 0])
        # f1.pts[:, 1] = image_processing.normalize(Kinv, f1.pts[:, 1])

        n_sample = len(f1.pts)

        #TODO: camera pose estimation + understand camera projection matrix
        if n_sample>8:
            try:
                model, inliers = ransac((f1.match[:, 0], f1.match[:, 1]), image_processing.EssentialMatrixTransform, min_samples=8, residual_threshold=0.02, max_trials=100)

                Rt, t = image_processing.fundamentalToRt(model.params)
                f1.pose = Rt.dot(f2.pose)

                # print(poses[-1])
                pts3D = image_processing.triangulate(f1.pose, f2.pose, f1.match)
                pts3D /= pts3D[:, 3:]

                v3D = []
                for i, pt in enumerate(pts3D):
                    # x, y = np.dot(K[:2, :2], pt[:2]) # f1.pose[:3, :3]
                    x, y, z = f1.pose[:3].dot(pt)
                    v3D.append([x, -y, -z])

                if (it%30)-20>0:
                    cloud_points.add_points(v3D)

            except Exception as e:
                print(e)
            
            if it % 30 == 0:
                if len(cloud_points.pointcloud)>1:
                    cloud_points.display_mesh()
                # cloud_points.set_points([])

        cv2.waitKey(1)
