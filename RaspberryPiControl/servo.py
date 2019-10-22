import wiringpi

# WARNING: this code only work with few pins as it does require the hardware PWM
# wPi pins: 1, 23, 24
# BCM pins: 18, 13, 19
# GPIO: 1, 23, 24
# Pysical pins: 8, 33, 35

class servo:
    "This is a servo motor class, use only wPi pins: 1, 23, 24; BCM pins: 18, 13, 19; GPIO: 1, 23, 24; Pysical pins: 8, 33, 35"
    def __init__(self, pin_num):
        self.__pin_num = pin_num
        wiringpi.wiringPiSetup()
        wiringpi.pinMode(self.__pin_num, wiringpi.PWM_OUTPUT)
        wiringpi.pwmSetMode(wiringpi.PWM_MODE_MS)
        wiringpi.pwmSetClock(384) #clock at 50kHz (20us tick)
        wiringpi.pwmSetRange(1000) #range at 1000 ticks (20ms)
        wiringpi.pwmWrite(self.__pin_num, 75) # theretically 50 (1ms) to 100 (2ms) on my servo 30-130 works ok
        self.min = 50
        self.max = 100
        print("servo init ok")

    def TurnLeft(self):
        wiringpi.pwmWrite(self.__pin_num, self.min)

    def TurnRight(self):
        wiringpi.pwmWrite(self.__pin_num, self.max)

    def MinMaxDuty(self, min, max):
        self.min = min
        self.max = max

