import getopt
import os
import sys


def printusage():
    print("python3 " + os.path.basename(__file__) + " -c <COM_Port>")
    print("<COM_Port>")
    print("  Windows: COMx where x is a number")
    print("  Linux: /dev/ttyXYZ where XYZ can be S2 or USB0 or any valid tty type")
    print("         keep in mind you need root priviledges to acces serial ports in Linux")
    print("         so execute the command in priviledge mode like: sudo python3 " +
          os.path.basename(__file__) + " -c /dev/ttyS2")
    print("  You can use as well environement variable COM_PORT. Arguments will prevent on environment variable")
    print("To install Xbox drivers on RPI: sudo apt-get install xboxdrv")


def get_port_name():
    comPort = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:h", ["com", "help"])
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

    if (comPort == ""):
        try:
            comPort = os.environ["COM_PORT"]
        except KeyError:
            printusage()
            sys.exit(2)
    return comPort


def get_controller_buttons(joy):
    direction = joy.leftX()
    throttle = joy.rightTrigger()
    brake = joy.leftTrigger()
    button_a = joy.A()
    button_x = joy.X()
    return (direction, throttle, brake, button_a, button_x)


def direction2categorical(direction, dir_range=(2, 10)):
    return round(dir_range[0]+direction*(dir_range[1] - dir_range[0]))


def dict2list(dict):
    return [i[1] for i in dict.items()]
