import cv2

cap = cv2.VideoCapture(0)
_, img = cap.read()

while(1):
    
    cv2.imshow('img',img)
    k = cv2.waitKey(33)

    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k) 
