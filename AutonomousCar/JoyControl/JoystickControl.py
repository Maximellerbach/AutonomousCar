from SerialCommand import control, motor, direction
import pygame
import sys, getopt, os

def printusage():
    print(__file__ + " -c <COM_Port>")
    print("  Windows: COMx where x is a number")
    print("  Linux: /dev/ttyXYZ where XYZ can be S2 or USB0")
    print("This code may not work on Raspberry PI, prefer to use the one with XBox native drivers.")


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

ser = SerialCommand.control(comPort)

pygame.init()

clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()

done = False

joystick_count = pygame.joystick.get_count()

while joystick_count ==0:
    clock.tick(20)

joystick = pygame.joystick.Joystick(0)
joystick.init()

while done==False:
    # EVENT PROCESSING STEP
    # for event in pygame.event.get(): # User did something
    #    if event.type == pygame.QUIT: # If user clicked close
    #        done=True # Flag that we are done so we exit this loop
    
    direction = joystick.get_axis(0)
    speed = joystick.get_axis(2)
    
    if direction < -0.2:
        # print("gauche")
        ser.ChangeDirection(3)    
    else if direction > 0.2:
        # print("droite")
        ser.ChangeDirection(11)
    else:
        # print("tout droit")
        ser.ChangeDirection(7)    

    if speed < -0.2:
        # print("acceleration")
        ser.ChangeMotorA(motor.MOTOR_FORWARD)        
        ser.ChangePWM(255 * speed)
    else if speed > 0.2:
        print("backward")
        ser.ChangeMotorA(motor.MOTOR_BACKWARD)
        ser.ChangePWM(255 * speed)
    else:
        ser.ChangePWM(0)

pygame.quit ()