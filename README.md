# Autonomous car project

The goal of this project is to participate to the iron car race.
TODO: add more context.

![car](/AutonomousCar/docs/thecar.jpg)
![car2](/AutonomousCar/docs/P_20190309_220020.jpg)

## The project

This project involve multiple aspects and is very rich:

- There is an Artificial Intellignece (AI) part to capture the road with a webcam (and potentially other sensors), analyze it, and take decisions. You will find more details in [this page](/AutonomousCar/ai.md). Language used for this part is Python 3.

- There is an Hardware part to be able to control the car. It does include batteries, power management, cooling, but as well choice of hardware to pilot the various elements like motors, servo motors. More [details here](/AutonomousCar/electronic.md).

![schema](/AutonomousCar/docs/schema.png)

- The board used for intelligence is a RockPro64 from pine64 running a Debian 9 (Stretch) arm64 v8 (aarch64) and as there are a lot of things to install on it, you'll find [the list of the main elements (openCV, Tensorflow, Numpy) and how to compile or install them here](/AutonomousCar/software.md).

![rockpro64](/AutonomousCar/docs/ROCKPro64_slide.jpg)

- Micro controler code, in our case an Arduino mini, to take orders from a board and actually pilot the low level electronic to make the motor move forward/backward as well as the car turning. [Documentation, protocole and code available](/ArduinoControl/readme.md). All the code for the Arduino is written in C.

![arduino](/AutonomousCar/docs/arduino.jpg)

- Classes and code to take the order from the AI and actually send it to the various electronic elements. More details can be [found in this section](/AutonomousCar/PythonSerialControl/readme.md). As for the rest of the high level code, it is written in Python, does involve Serial Port communications.

- Classes and apps including [test tools to remotely pilot the car in wifi for example](/AutonomousCar/PythonSerialControl/Webcontrol.py) and [previous versions involving controloing the electronic directly from a Raspberry Pi](/AutonomousCar/RaspberryPiControl/readme.md). All written in Python.

- Classes and apps to [test image capture](/AutonomousCar/ImageWeb). All written in Python.
