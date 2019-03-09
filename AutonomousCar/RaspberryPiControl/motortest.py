import motor
import time

mot = motor.motor(21, 22, 23)

while True:
    mot.speed(100)
    for i in range(0, 10):
        mot.forward()
        print("forward")
        time.sleep(2.0)
        mot.stop()
        print("stop")
        time.sleep(2.0)
        mot.backward()
        print("backward")
        time.sleep(2.0)
        mot.idle()
        print("idle")
        time.sleep(2.0)
    # for i in range(0,100):
    #    mot.speed(i)
    #    time.sleep(0.2)
    
        
        
