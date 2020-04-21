import numpy as np
from glob import glob
import cv2

class data(): # TODO: clean data class (could be used elsewhere)
    def __init__(self, dos, is_float=True):
        self.dos = dos
        self.is_float = is_float

    def load_lab(self):
        X_lab = []
        for path in glob(self.dos+"*"):
            print(path)
            lab = path.split('\\')[-1].split('_')[0]
            if self.is_float:
                lab = float(lab)
            else:
                lab = int(lab)
                lab = self.transform_lab(lab)

            X_lab.append([path, lab])
        return X_lab

    def transform_lab(self, lab, dico=[3, 5, 7, 9, 11]):
        return (dico.index(lab)-2)/2

    def average_data(self, X_lab, index=1, window_size=10):
        averaged = []
        for i in range(window_size//2, len(X_lab)-window_size//2):
            averaged.append(np.average(X_lab[i-window_size//2: i+window_size//2, 1], axis=-1))

        for i in range(len(X_lab)):
            if i<=window_size//2 or i>=len(X_lab)-window_size//2:
                del X_lab[i]
            else:
                X_lab[i, index] = averaged[i]
            

class make_labels(data): # TODO: look further in the data to make labels
    def __init__(self, dos="C:\\Users\\maxim\\recorded_imgs\\0\\"):
        super().__init__(dos, is_float=False)
        self.X_lab = self.load_lab()
        self.average_data(self.X_lab, window_size=10)
        
        print(len(self.X_lab))


if __name__ == "__main__":
    labs = make_labels(dos="")