import numpy as np
import cv2
import matplotlib.pyplot as plt
import reorder_dataset

class data(): # TODO: clean data class (could be used elsewhere)
    def __init__(self, dos, is_float=True):
        self.dos = dos
        self.is_float = is_float

    def load_lab(self):
        X = []
        labs = []
        dts, datalen = reorder_dataset.load_dataset(self.dos, recursive=False)
        for path in dts:
            lab = path.split('\\')[-1].split('_')[0]
            if self.is_float:
                lab = float(lab)
            else:
                lab = int(lab)
                lab = self.transform_lab(lab)

            X.append(path)
            labs.append(lab)
        return np.array(X), np.array(labs)

    def transform_lab(self, lab, dico=[3, 5, 7, 9, 11]):
        return (dico.index(lab)-2)/2

    def average_data(self, X, labs, window_size=10):
        averaged = []
        for i in range(window_size//2, len(labs)-window_size//2):
            averaged.append(np.average(labs[i-window_size//2: i+window_size//2], axis=-1))

        index_modifier = 0
        labs[window_size//2:-window_size//2] = averaged

        return X, labs

    def detect_spike(self, labs, th=0.5, window_size=10):
        spikes = []
        spike = []
        is_spike = False
        for it, lab in enumerate(labs):
            if lab>=th and is_spike == False:
                spike.append(it-window_size//2)
                is_spike = True

            elif lab<th and is_spike == True:
                spike.append(it+window_size//2)
                is_spike = False
                spikes.append(spike)
                spike = []

        return spikes

    def get_timetoclosestturn(self, X, spikes):
        return
            

class make_labels(data): # TODO: look further in the data to make labels
    def __init__(self, dos="C:\\Users\\maxim\\recorded_imgs\\0\\", is_float=True):
        super().__init__(dos, is_float=is_float)
        window_size = 20

        self.X, self.labs = self.load_lab()
        original_labs = self.labs

        self.X, self.labs = self.average_data(self.X, self.labs, window_size=window_size)
        self.detect_spike(self.labs, th=0.5, window_size=window_size)

        plt.plot([i for i in range(len(self.labs))], self.labs, linewidth=1)
        plt.plot([i for i in range(len(original_labs))], original_labs, linewidth=1)

        plt.show()


if __name__ == "__main__":
    labs = make_labels(dos="C:\\Users\\maxim\\recorded_imgs\\0_0_1587469658.7638354\\", is_float=True)