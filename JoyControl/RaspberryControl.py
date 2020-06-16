import getopt
import os
import sys
import time

import cv2
import xbox

sys.path.append('../custom_modules/')
from SerialCommand import control, motor

MAXSPEED = 120
dico_save = [3,5,7,9,11]
dico = [10,8,6,4,2]

def create_save_string(components, path="../../image_raw/"):
    path = ""
    len_cmp = len(components)
    for it, comp in enumerate(components):
        path += str(comp)
        if it != len_cmp:
            path += "_"
        else:
            path += ".png"
    return path

def printusage():
    print(__file__ + " -c <COM_Port>")
    print("  Windows: COMx where x is a number")
    print("  Linux: /dev/ttyXYZ where XYZ can be S2 or USB0")
    print("To install Xbox drivers on RPI: sudo apt-get install xboxdrv")


comPort = ""
try:
    opts, args = getopt.getopt(sys.argv[1:],"c:h",["com", "help"])
except getopt.GetoptError:
    printusage()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-h", "--help"):
        printusage()
        sys.exit()
    elif opt in ("-c", "--com"):
        comPort = arg
    else:
        printusage()
        sys.exit(2)
# if (opts == []):
#     printusage()
#     sys.exit(2)
if (comPort == ""):
    try:
        comPort = os.environ["COM_PORT"]
    except KeyError: 
        pass

# Give full access to the serial port
# Called "mode bourrin" in French :-)
os.system('sudo chmod 0666 ' + comPort)
ser = control(comPort)
joy = xbox.Joystick()

cap = cv2.VideoCapture(0)
cat_mode = False
while not joy.Back():

    direction = joy.leftX()
    throttle = joy.rightTrigger()
    throttle2 = joy.leftTrigger()
    thethrottle = throttle - throttle2
    rec = joy.A()
    do_dir = joy.X()
    
    if cat_mode:
        if direction < -0.4:
            direc= 0

        elif direction > 0.4:
            direc= 4

        elif direction < -0.2:
            direc= 1

        elif direction > 0.2:
            direc= 3

        else:
            direc= 2

        if not do_dir:
            ser.ChangeDirection(dico[direc])
    else:
        th_direction = 0.05
        if abs(direction) > abs(th_direction):
            direc = direction
        else:
            direc = 0

    
    if abs(thethrottle) > 0.2:
        #print("increase speed")  
        ser.ChangeMotorA(motor.MOTOR_BACKWARD)
        pwm = int(MAXSPEED * thethrottle)
        ser.ChangePWM(pwm)

    else:
        ser.ChangePWM(0)

    if rec:
        #read cam and resize the image
        _, img= cap.read()
        img= cv2.resize(img,(160,120))
        
        
        if cat_mode:
            # save it with for label, the joystick values mapped on 5 directions
            save_string = create_save_string([dico_save[direc], time.time()])
            cv2.imwrite(save_string, img)
        else:
            save_string = create_save_string([direc, thethrottle, time.time()])
            cv2.imwrite(save_string, img)


joy.close()
print('terminated')

