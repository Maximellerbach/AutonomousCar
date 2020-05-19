import SerialCommand
import time
import sys, getopt, os

def printusage():
    print(__file__ + " -c <COM_Port>")
    print("  Windows: COMx where x is a number")
    print("  Linux: /dev/ttyXYZ where XYZ can be S2 or USB0")


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
        comPort = arg.strip()
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
    if (comPort!=""):
        ser = SerialCommand.control(comPort)
        try:
            print("Turns:" + str(ser.GetTurns()))

            print("start changing PWM from 0 to 255")
            for pwm in range(0,255):
                ser.ChangePWM(pwm)
                time.sleep(0.2)
                print("Turns:" + str(ser.GetTurns()))
            print("finished changing PWM")

            print("Turns:" + str(ser.GetTurns()))

            for dir in SerialCommand.direction:
                ser.ChangeDirection(dir)
                print("changed direction " + str(dir))
                time.sleep(5)
                print("Turns:" + str(ser.GetTurns()))

            for motor in SerialCommand.motor:
                ser.ChangeMotorA(motor)
                print("changed motor A " + str(motor))
                time.sleep(5)
                print("Turns:" + str(ser.GetTurns()))

            for motor in SerialCommand.motor:
                ser.ChangeMotorB(motor)
                print("changed motor B " + str(motor))
                time.sleep(5)
                print("Turns:" + str(ser.GetTurns()))

            ser.ChangeAll(SerialCommand.direction.DIR_STRAIGHT, SerialCommand.motor.MOTOR_STOP, SerialCommand.motor.MOTOR_STOP, 255)
            print("changed all")

            print("Turns:" + str(ser.GetTurns()))
        except KeyboardInterrupt:
            ser.__exit__()

    else:
        printusage()


