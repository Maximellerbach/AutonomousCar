import xbox
from SerialCommand import control, motor, direction
import sys, getopt, os
import cv2
import time

MAXSPEED = 120

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
    
    if direction < -0.4:
        # print("full left")
        direc=3
        ser.ChangeDirection(11)       
    elif direction > 0.4:
        # print("full right")
        direc= 11
        ser.ChangeDirection(2)
    elif direction < -0.2:
        # print("left")
        direc= 5
        ser.ChangeDirection(9)
    elif direction > 0.2:
        # print("right")
        direc= 9
        ser.ChangeDirection(4)
    else:
        direc=7
        ser.ChangeDirection(6)
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
    
    #read cam and resize the image
    _, img= cap.read()
    img= cv2.resize(img,(160,120))

    #save it with for label, the joystick values mapped on 5 directions
    cv2.imwrite('../../image_raw/'+str(direc)+'_'+str(time.time())+'.png',img)

joy.close()

print('terminated')


