# Autonomous car project

then: 
![car](/docs_image/car.jpg)

and now: 
![car2](/docs_image/car2.jpg)

## The project

This project involve multiple aspects and is very rich:

- There is an Artificial Intelligence (AI) part to capture the road with a webcam (and potentially other sensors), analyze it, and take decisions. You will find more details in [this page](/docs/ai.md). Language used for this part is Python 3.6.8

- There is an Hardware part to be able to control the car. It does include batteries, power management, cooling, but as well choice of hardware to pilot the various elements like motors, servo motors. More [details here](/docs/electronic.md).

The old hardware:
- The board used for intelligence is a RockPro64 from pine64 running a Debian 9 (Stretch) arm64 v8 (aarch64) and as there are a lot of things to install on it, you'll find [the list of the main elements (openCV, Tensorflow, Numpy) and how to compile or install them here](/docs/software.md).
![rockpro64](/docs_image/ROCKPro64_slide.jpg)
![schema](/docs_image/schema.png)

New hardware:
- The board I use now is a Raspberry Pi 3b+, running raspbian, you will find some tutorials on how to install Tensorflow and OpenCV on internet.
![schema2](/docs_image/schema2.png)


- Micro controler code, in our case an Arduino mini, to take orders from a board and actually pilot the low level electronic to make the motor move forward/backward as well as the car turning. [Documentation, protocole and code available](/ArduinoControl/readme.md). All the code for the Arduino is written in C.

![arduino](/docs_image/arduino.jpg)

- Classes and code to take the order from the AI and actually send it to the various electronic elements. More details can be [found in this section](/python_serial_control/readme.md). As for the rest of the high level code, it is written in Python, does involve Serial Port communications.

- Classes and apps including [test tools to remotely pilot the car in wifi for example](/python_serial_control/web_control.py) and [previous versions involving controling the electronic directly from a Raspberry Pi](/RaspberryPiControl/readme.md). All written in Python.

- Classes and apps to [test image capture](/ImageWeb). All written in Python.