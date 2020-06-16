import getopt
import os
import platform
import sys

from flask import Flask

sys.path.append('../custom_modules/')
from SerialCommand import control, direction, motor


OK = "ok"
ERROR = "error"

def printusage():
    print("python3 " + os.path.basename(__file__) + " -c <COM_Port>")
    print("<COM_Port>")
    print("  Windows: COMx where x is a number")
    print("  Linux: /dev/ttyXYZ where XYZ can be S2 or USB0 or any valid tty type")
    print("         keep in mind you need root priviledges to acces serial ports in Linux")
    print("         so execute the command in priviledge mode like: sudo python3 " + os.path.basename(__file__) + " -c /dev/ttyS2")
    print("  You can use as well environement variable COM_PORT. Arguments will prevent on environment variable")

def getPortName(argv):
    #plat = platform.system()
    #if (plat == "Windows"):
    #    ser = control("COM4")
    #else:
    #    ser = control("/dev/ttyS2")
    #    ser = control("/dev/ttyUSB0")
    comPort = ""
    try:
        opts, args = getopt.getopt(argv,"c:h",["com=", "help"])
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
    if (comPort == ""):
        try:
            comPort = os.environ["COM_PORT"]
        except KeyError: 
            printusage()
            sys.exit(2)
    return comPort

app = Flask(__name__)

ser = control(getPortName(sys.argv[1:]))

def cleanargs(val):
    retval = str(val).split('?')
    return retval[0]

@app.route('/')
def hello():
    return app.send_static_file("index.html")

@app.route('/motor/<name>/<dir>')
def pilotmotor(name, dir):
    val = int(cleanargs(dir))
    mot = str(name).lower()
    if val in range(0, len(motor)-1):
        if (mot == "a"):
            ser.ChangeMotorA(val)
        else: 
            if (mot == "b"):
                ser.ChangeMotorB(val)
            else:
                return ERROR
        return OK
    return ERROR

@app.route('/dir/<dir>')
def pilotdir(dir):
    val = int(cleanargs(dir))
    if (val in range(0, len(direction)-1)):
        ser.ChangeDirection(val)
        return OK
    return ERROR

@app.route('/pwm/<pwm>')
def pilotpwm(pwm):
    val = int(cleanargs(pwm))
    if (val in range(0, 255)):
        ser.ChangePWM(val)
        return OK
    return ERROR

@app.route('/stop')
def pilotstop():
    ser.ChangeAll(direction.DIR_STRAIGHT,motor.MOTOR_STOP,motor.MOTOR_STOP,0)
    return OK

@app.route('/turns')
def turns():
    out = ser.GetTurns()
    return str(out)

if __name__ == '__main__':

    # run flask, host = 0.0.0.0 needed to get access to it outside of the host
    app.run(host='0.0.0.0',port=1337)
