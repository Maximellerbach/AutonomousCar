import numpy as np
import autolib
import keras
import cv2
from glob import glob

class image_generator(keras.utils.Sequence):
    def __init__(self, img_path, datalen, batch_size=32, augm=True, proportion=0.15, cat=True, smoothing=0.1, label_rdm=0, shape=(160,120,3), n_classes=5, memory=49, seq=False, reconstruction=False):
        self.shape = shape
        self.augm = augm
        self.img_cols = shape[0]
        self.img_rows = shape[1]
        self.batch_size = batch_size
        self.img_path = img_path
        self.n_classes = n_classes
        self.memory_size = memory+1
        self.datalen = datalen
        self.seq = seq
        self.reconstruction = reconstruction
        self.proportion = proportion
        self.smoothing = smoothing
        self.label_rdm = label_rdm
        self.cat = cat


    def __data_generation(self, img_path):
        batchfiles = np.random.choice(img_path, size=self.batch_size)
        xbatch = []
        ybatch = []

        for i in batchfiles:
            try:
                img = cv2.imread(i)
                img = cv2.resize(img, (self.img_cols, self.img_rows))
                # img = autolib.cut_img(img, 20)

                label = autolib.get_label(i, flip=False, before=True, reg=False) #reg=True for [-1;1] directions

                xbatch.append(img)
                ybatch.append(label[0])
            except:
                print(i)


        if self.augm == True:
            X_bright, Y_bright = autolib.generate_brightness(xbatch, ybatch, proportion=self.proportion)
            X_gamma, Y_gamma = autolib.generate_low_gamma(xbatch, ybatch, proportion=self.proportion)
            X_night, Y_night = autolib.generate_night_effect(xbatch, ybatch, proportion=self.proportion)
            X_shadow, Y_shadow = autolib.generate_random_shadows(xbatch, ybatch, proportion=self.proportion)
            X_chain, Y_chain = autolib.generate_chained_transformations(xbatch, ybatch, proportion=self.proportion)
            X_noise, Y_noise = autolib.generate_random_noise(xbatch, ybatch, proportion=self.proportion)
            X_rev, Y_rev = autolib.generate_inversed_color(xbatch, ybatch, proportion=self.proportion)
            X_glow, Y_glow = autolib.generate_random_glow(xbatch, ybatch, proportion=self.proportion)
            X_cut, Y_cut = autolib.generate_random_cut(xbatch, ybatch, proportion=self.proportion)

            not_emptyX = [i for i in (xbatch, X_gamma, X_bright, X_night, X_shadow, X_chain, X_noise, X_rev, X_glow, X_cut) if len(i)!=0]
            not_emptyY = [i for i in (ybatch, Y_gamma, Y_bright, Y_night, Y_shadow, Y_chain, Y_noise, Y_rev, Y_glow, Y_cut) if len(i)!=0]

            xbatch = np.concatenate(not_emptyX)
            ybatch = np.concatenate(not_emptyY)
        
        xflip, yflip = autolib.generate_horizontal_flip(xbatch, ybatch, proportion=1)
        xbatch = np.concatenate((xbatch, xflip))/255
        ybatch = np.concatenate((ybatch, yflip))

        # ybatch = to_categorical(ybatch, self.n_classes)
        if self.cat == True: #TODO: do autolib function for -1 / 1 range
            ybatch = autolib.label_smoothing(ybatch, self.n_classes, self.smoothing, random=self.label_rdm)
        

        return xbatch, ybatch

    def __data_generationseq(self, dos_path, memory_size):
        fold = np.random.randint(0, len(dos_path), size=self.batch_size)
        batchfiles = []
        for f in fold:
            avpath = dos_path[f]
            x = np.random.randint(1, len(avpath))
            if x-memory_size>0:
                paths = avpath[x-memory_size:x]
            else:
                paths = ["somekind\\"+str(np.random.choice([3, 5, 7, 9, 11]))+"_ofdummydata.png"]*np.absolute(x-memory_size)+avpath[:x]
            batchfiles.append(paths)

        x1batch = []
        x2batch = []
        y1batch = []
        y2batch = []
        ys1batch = []
        ys2batch = []


        for i in batchfiles:
            img = cv2.imread(i[-1])
            try:
                img = cv2.resize(img, (self.img_cols, self.img_rows))
                lab, rev = autolib.get_previous_label(i)

                x1batch.append(img)
                x2batch.append(cv2.flip(img, 1))
                ys1batch.append(lab[:-1])
                ys2batch.append(rev[:-1])
                y1batch.append(lab[-1])
                y2batch.append(lab[-1])
            except:
                print(i)

        xbatch = np.concatenate((x1batch, x2batch))
        ybatch = np.concatenate((y1batch, y2batch))
        ysbatch = np.concatenate((ys1batch, ys2batch))
        
        if self.augm == True:
            X_bright, Y_bright, ys_b = autolib.generate_brightness(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_gamma, Y_gamma, ys_g = autolib.generate_low_gamma(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_night, Y_night, ys_n = autolib.generate_night_effect(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_shadow, Y_shadow, ys_s = autolib.generate_random_shadows(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_chain, Y_chain, ys_c = autolib.generate_chained_transformations(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_noise, Y_noise, ys_rdm = autolib.generate_random_noise(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_rev, Y_rev, ys_r = autolib.generate_inversed_color(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_glow, Y_glow, ys_glow = autolib.generate_random_glow(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            X_cut, Y_cut, ys_cut = autolib.generate_random_cut(xbatch, ybatch, ys=ysbatch, proportion=self.proportion, ysb=True)
            
            not_emptyX = [i for i in (xbatch, X_gamma, X_bright, X_night, X_shadow, X_chain, X_noise, X_rev, X_glow, X_cut) if len(i)!=0]
            not_emptyY = [i for i in (ybatch, Y_gamma, Y_bright, Y_night, Y_shadow, Y_chain, Y_noise, Y_rev, Y_glow, Y_cut) if len(i)!=0]
            not_emptyYs = [i for i in (ysbatch, ys_b, ys_g, ys_n, ys_s, ys_c, ys_rdm, ys_r, ys_glow, ys_cut) if len(i)!=0]

            xbatch = np.concatenate(not_emptyX)/255
            ybatch = np.concatenate(not_emptyY)
            ysbatch = np.concatenate(not_emptyYs)

        else:
            xbatch = xbatch/255

        # ybatch = to_categorical(ybatch, self.n_classes)
        # ybatch = autolib.label_smoothing(ybatch, self.n_classes, self.smoothing, random=self.label_rdm)

        yss = []
        for ysb in ysbatch:
            # yss.append(autolib.label_smoothing(ysb, 5, 0.25))
            
            if self.cat == True: #TODO: do autolib function for -1 / 1 range
                ysbatch = autolib.label_smoothing(ysb, self.n_classes, self.smoothing, random=self.label_rdm)
        
            yss.append(ysbatch)
        yss = np.array(yss)
        return xbatch, yss, ybatch

    def __data_generation_reconstruction(self, img_path):
        batchfiles = np.random.choice(img_path, size=self.batch_size)
        xbatch = []
        ybatch = []

        for i in batchfiles:
            try:
                img = cv2.imread(i)
                img = cv2.resize(img, (self.img_cols, self.img_rows))
                # img = autolib.cut_img(img, 20)

                label = autolib.get_label(i, flip=False, before=True, reg=False) #reg=True for [-1;1] directions

                xbatch.append(img)
                ybatch.append(label[0])
            except:
                print(i)

        xflip, yflip = autolib.generate_horizontal_flip(xbatch, ybatch, proportion=1)
        xbatch = np.concatenate((xbatch, xflip))
        ybatch = np.concatenate((ybatch, yflip))

        if self.augm == True:
            X_bright, _ = autolib.generate_brightness(xbatch, ybatch, proportion=self.proportion)
            X_gamma, _ = autolib.generate_low_gamma(xbatch, ybatch, proportion=self.proportion)
            X_night, _ = autolib.generate_night_effect(xbatch, ybatch, proportion=self.proportion)
            X_shadow, _ = autolib.generate_random_shadows(xbatch, ybatch, proportion=self.proportion)
            X_chain, _ = autolib.generate_chained_transformations(xbatch, ybatch, proportion=self.proportion)
            X_noise, _ = autolib.generate_random_noise(xbatch, ybatch, proportion=self.proportion)
            X_rev, _ = autolib.generate_inversed_color(xbatch, ybatch, proportion=self.proportion)
            X_glow, _ = autolib.generate_random_glow(xbatch, ybatch, proportion=self.proportion)
            X_cut, _ = autolib.generate_random_cut(xbatch, ybatch, proportion=self.proportion)

            not_emptyX = [i for i in (xbatch, X_gamma, X_bright, X_night, X_shadow, X_chain, X_noise, X_rev, X_glow, X_cut) if len(i)!=0]
            xbatch = np.concatenate(not_emptyX)/255

        else:
            xbatch = xbatch/255

        return xbatch

    def __len__(self):
        return int(self.datalen/self.batch_size)

    def __getitem__(self, index):
        if self.reconstruction==True:
            X = self.__data_generation_reconstruction(self.img_path)
            return X, X

        elif self.seq==True:
            X1, X2, Y = self.__data_generationseq(self.img_path, self.memory_size)
            return [X1, X2], Y
        
        else:
            X, Y = self.__data_generation(self.img_path)
            return X, Y


