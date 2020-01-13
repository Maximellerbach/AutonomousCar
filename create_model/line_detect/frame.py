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

    def to_array(self):
        self.kps = np.array(self.kps)
        self.des = np.array(self.des)
        self.idx1 = np.array(self.idx1)
        self.idx2 = np.array(self.idx2)
        self.pts = np.array(self.pts)
        self.matches = np.array(self.matches)
