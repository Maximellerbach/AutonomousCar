import cv2
import numpy as np
import math

def get_diff(img, prev, show=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    diff = np.absolute(gray-prev*0.99)
    diff = np.array(diff, dtype = 'uint8')

    if show == True:
        cv2.imshow('dif', diff/255)

    return diff, gray


def triangulate(pose1, pose2, match):
    ret = np.zeros((match.shape[0], 4))

    for i, p in enumerate(match):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret

def fundamentalToRt(F):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]

    # TODO: Resolve ambiguities in better ways. This is wrong.
    if t[2] < 0:
        t *= -1
    
    return np.linalg.inv(poseRt(R, t)), t

def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

def get_kps_distance(kp1, kp2):
    x1, y1 = kp1.pt
    x2, y2 = kp2.pt
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def get_arr_distance(arr, arr2):
    return (np.power(arr, 2) + np.power(arr2, 2))

def get_vect(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2

    return ((x1-x2), (y1-y2))


def matches2vec(matches, kps1, kps2):
    vectors = []
    for match in matches:
        q= match.queryIdx
        t= match.trainIdx

        x1, y1 = kps1[q].pt
        x2, y2 = kps2[t].pt

        v = ((x2-x1), (y2-y1))
        vectors.append(v)
    return vectors

def empty_grid(y, x, item=(0.,0.)):
    g = []
    for i in range(y):
        h = []
        for j in range(x):
            h.append([item])
        g.append(h)
    return g

def capture_img(cap, res, show=True):
    _, img = cap.read()
    img = cv2.resize(img, (res[1],res[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if show == True:
        cv2.imshow("img", img)

    return img, gray


def extractFeatures(orb, img, cpimg, show=True):
    # detection
    pts = cv2.goodFeaturesToTrack(img, 1500, qualityLevel=0.01, minDistance=8)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=32) for f in pts]
    kps, des = orb.compute(img, kps)
  
    if show == True:
        cpimg = cv2.drawKeypoints(cpimg, kps, cpimg, color=(0,255,0), flags=0)

        cv2.imshow('orb', cpimg/255)

    # return pts and des
    return kps, des

def normalize(Kinv, pts, n=3):
    if n == 3:
        return np.dot(Kinv, pts).T[:, 0:2]
    else:
        return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x,np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def to_gridvec(pts1, pts2, img_shape, grid_shape=(16,16), vec3D=False):
    h, w = img_shape
    hg, wg = grid_shape
    hdiv = int(h/hg)+1
    wdiv = int(w/wg)+1

    if vec3D ==True:
        item = (0., 0., 0.)
    else:
        item = (0., 0.)
    grid = empty_grid(hg, wg, item=item)
    # print(np.array(grid).shape)

    for i in range(len(pts1)):
        p1 = pts1[i]
        p2 = pts2[i]
        
        xv, yv = get_vect(p1, p2)
        if vec3D==True:
            v = (xv*w, yv*h, 0)
        else:
            v = (xv*w, yv*h)

        xg = int(p1[0]*w//wdiv)
        yg = int(p1[1]*h//hdiv)
        # print(yg, xg)
        grid[yg][xg].append(v)

    return grid

class EssentialMatrixTransform(object):
    def __init__(self):
        self.params = np.eye(3)

    def __call__(self, coords):
        coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
        return coords_homogeneous @ self.params.T

    def estimate(self, src, dst):
        assert src.shape == dst.shape

        if src.shape[0] >= 8:
            # Setup homogeneous linear equation as dst' * F * src = 0.
            A = np.ones((src.shape[0], 9))
            A[:, :2] = src
            A[:, :3] *= dst[:, 0, np.newaxis]
            A[:, 3:5] = src
            A[:, 3:6] *= dst[:, 1, np.newaxis]
            A[:, 6:8] = src

            # Solve for the nullspace of the constraint matrix.
            _, _, V = np.linalg.svd(A)
            F = V[-1, :].reshape(3, 3)

            # Enforcing the internal constraint that two singular values must be
            # non-zero and one must be zero.
            U, S, V = np.linalg.svd(F)
            S[0] = S[1] = (S[0] + S[1]) / 2.0
            S[2] = 0
            self.params = U @ np.diag(S) @ V

            return True
        else:
            return False
        
    def residuals(self, src, dst):
        # Compute the Sampson distance.
        src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
        dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])

        F_src = self.params @ src_homogeneous.T
        Ft_dst = self.params.T @ dst_homogeneous.T

        dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

        return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2
                                        + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)

