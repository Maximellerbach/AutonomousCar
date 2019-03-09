import cv2
import numpy as np

w = 160
h = 120

low = np.array([10,100,60])
up = np.array([20,255,255])

l = np.array([0, 0, 150])
u = np.array([255, 20, 255])

img = cv2.imread('/home/pi/Downloads/img.png')
img = cv2.resize(img,(w,h))
img = img[int(h/3):h, 0:w]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, l, u)+cv2.inRange(hsv,low,up)


img = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow('img',img)
cv2.waitKey(0)
