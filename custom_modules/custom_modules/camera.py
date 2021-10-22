from threading import Thread

import cv2
import time


class usbWebcam():
    def __init__(self, device=0, topcrop=0.20, botcrop=0.0):
        self.cap = cv2.VideoCapture(device)

        self.last_image = None
        self.new_image = False

        img_shape = self.cap.read()[1].shape
        self.h = img_shape[0]
        self.croptop = int(topcrop * img_shape[0])
        self.cropbot = img_shape[0] - int(img_shape[0] * botcrop)

        self.running = True

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            ret, img = self.cap.read()
            if ret is False:
                self.running = False
            else:
                self.last_image = self.crop_image(img)
                self.new_image = True

    def read(self):
        if self.running:
            if self.new_image:
                self.new_image = False
                return self.last_image
            else:
                while self.new_image is False and self.running:
                    time.sleep(0.001)

                self.new_image = False
                return self.last_image
        else:
            raise Exception("Is the camera connected?")

    def release(self):
        self.running = False
        self.cap.release()

    def crop_image(self, img):
        return img[int(self.croptop*self.h): self.h-int(self.h*self.cropbot)].copy()


if __name__ == "__main__":
    cap = usbWebcam()
    cap.start()
    while(True):
        cam = cap.read()
        cam = cv2.resize(cam, (160, 120))

        cv2.imshow('cam', cam)
        cv2.waitKey(1)
