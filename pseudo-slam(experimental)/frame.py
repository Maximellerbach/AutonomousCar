import numpy as np

class Frame():
    def __init__(self):
        self.kps = []
        self.des = []
        self.idx1 = []
        self.idx2 = []
        self.pts = []
        self.matches = []
        self.pose = np.eye(4)

        self.points = []
        self.ordered = []
        self.match = []

    def to_array(self):
        self.kps = np.array(self.kps)
        self.des = np.array(self.des)
        self.idx1 = np.array(self.idx1)
        self.idx2 = np.array(self.idx2)
        self.pts = np.array(self.pts)
        self.matches = np.array(self.matches)

def get_points(f, n=2):
    p = [i.pt[:n] for i in f.points]
    return np.array(p)

def set_points(f, pts):
    for it, i in enumerate(f.points):
        i.pt = pts[it]

def get_matching(f1, f2, n=2, img_shape=(940, 560)):
    match = []
    for i in f1.points:
        p = list(f2.kps[i.index].pt)
        p[0] = p[0]/img_shape[0]
        p[1] = p[1]/img_shape[1]

        match.append((i.pt[:n], p))
    f1.match = np.array(match)