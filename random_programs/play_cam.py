import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    while True:
        _, img = cap.read()
        cv2.imshow("cam", img)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
