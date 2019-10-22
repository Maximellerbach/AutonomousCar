import xbox
from SerialCommand import control, motor, direction
import sys, getopt, os
import cv2
import time

MAXSPEED = 120
dico_save = [3,5,7,9,11]
dico = [10,8,6,4,2]

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

done = False

cap= cv2.VideoCapture(0)

while not joy.Back():
    
    direction = joy.leftX()
    speed = joy.rightTrigger()
    speed2 = joy.leftTrigger()
    thespeed = speed - speed2
    rec = joy.A()
    do_dir = joy.X()
    
    
    if direction < -0.4:
        direc=0

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

    
    if thespeed > 0.2:
        #print("increase speed")  
        ser.ChangeMotorA(motor.MOTOR_BACKWARD)
        pwm = int(MAXSPEED * thespeed)
        ser.ChangePWM(pwm)

    elif thespeed < -0.2:
        #print("backward")
        ser.ChangeMotorA(motor.MOTOR_FORWARD)
        pwm = -int(MAXSPEED * thespeed)
        ser.ChangePWM(pwm)

    else:
        ser.ChangePWM(0)

    if rec:
        #read cam and resize the image
        _, img= cap.read()
        img= cv2.resize(img,(160,120))

        
        #save it with for label, the joystick values mapped on 5 directions
        cv2.imwrite('../../image_raw/'+str(dico_save[direc])+'_'+str(time.time())+'.png',img)

    '''

    if abs(direction)>0.05:
        direc = direction
    else:
        direc = 0 

    if rec:
        #read cam and resize the image
        _, img= cap.read()
        img= cv2.resize(img,(160,120))

        cv2.imwrite('../../image_reg/'+str(direc)+'_'+str(time.time())+'.png', img)
    '''


joy.close()
print('terminated')


