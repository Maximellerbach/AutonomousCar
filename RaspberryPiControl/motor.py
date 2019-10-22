import wiringpi

class motor:
    "This class is intended to pilot a hbridge with or without a PWM. For PWM, use only wPi pins: 1, 23, 24; BCM pins: 18, 13, 19; GPIO: 1, 23, 24; Pysical pins: 8, 33, 35"
    def __init__(self, pin_forward, pin_backward, pin_pwm):
        self.__pin_forward = pin_forward
        self.__pin_backward = pin_backward
        self.__pin_pwm = pin_pwm
        wiringpi.wiringPiSetup()
        wiringpi.pinMode(self.__pin_forward, wiringpi.OUTPUT)
        wiringpi.pinMode(self.__pin_backward, wiringpi.OUTPUT)
        if (self.__pin_pwm >0 ):
            wiringpi.pinMode(self.__pin_pwm, wiringpi.PWM_OUTPUT)
            wiringpi.pwmSetMode(wiringpi.PWM_MODE_MS)
            wiringpi.pwmSetClock(500) # 500 = 19kHz. If the motor uses under 2 A and the PWM frequency is held between 9 KHz and 25 KHz it is ok. 
            wiringpi.pwmSetRange(100) # we will use a range of *0 -> 100 while the default range which is 1024
            wiringpi.pwmWrite(self.__pin_pwm, 0)
        wiringpi.digitalWrite(self.__pin_forward, 0)
        wiringpi.digitalWrite(self.__pin_backward, 0)

    def speed(self, speed):
        if (speed < 0):
            speed = 0
        if (speed > 100):
            speed = 100
        self.__speed = speed
        if (self.__pin_pwm > 0):
            wiringpi.pwmWrite(self.__pin_pwm, speed)

    def getSpeed(self):
        return self.__speed

    def stop(self):
        if (self.__pin_pwm > 0):
            wiringpi.pwmWrite(self.__pin_pwm, 0)
        wiringpi.digitalWrite(self.__pin_forward, 1)
        wiringpi.digitalWrite(self.__pin_backward, 1)      

    def idle(self):
        wiringpi.pwmWrite(self.__pin_pwm, 0)
        wiringpi.digitalWrite(self.__pin_forward, 0)
        wiringpi.digitalWrite(self.__pin_backward, 0)  

    def forward(self):
        wiringpi.digitalWrite(self.__pin_forward, 1)
        wiringpi.digitalWrite(self.__pin_backward, 0)  

    def backward(self):
        wiringpi.digitalWrite(self.__pin_forward, 0)
        wiringpi.digitalWrite(self.__pin_backward, 1) 