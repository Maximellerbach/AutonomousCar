import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while(True):
        _, img = cap.read()
        cv2.imshow('cam', img)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
