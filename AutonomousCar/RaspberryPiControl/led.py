import time
import wiringpi

pinnum=23
# One of the following MUST be called before using IO functions:
wiringpi.wiringPiSetup()     # For sequential pin numbering
print("setup ok")

wiringpi.pinMode(pinnum, 1)
print("pinmode ok")

while True:
    wiringpi.digitalWrite(pinnum, 1)
    time.sleep(1.0)
    wiringpi.digitalWrite(pinnum, 0)
    time.sleep(1.0)