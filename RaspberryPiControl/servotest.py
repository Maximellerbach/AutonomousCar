import time
import servo

myservp = servo.servo(1)
# Main loop will run forever moving between 1.0 and 2.0 mS long pulses:
while True:
    #servo.duty_cycle = servo_duty_cycle(1.0)
    #time.sleep(1.0)
    #servo.duty_cycle = servo_duty_cycle(2.0)
    #time.sleep(1.0)
    myservp.TurnLeft();
    time.sleep(1.0)
    myservp.TurnRight()
    time.sleep(1.0)
    