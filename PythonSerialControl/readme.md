# Protocol and usage or serial command send to an Arduino to pilot an autonomous car

This code allow to pilot thru serial port an autonomous car thru serial commands. This file describes the protocol as well as the Python implementation and the Arduino implementation.

Please check [this page](webcontrol.md) if you want to test the protocole in a web page.

## The communication protocol

In the current implementation, the protocole is detined the following way to be efficient and simple:

| bit | Function | Addressable field |
| --- | --- | --- |
| 0 | direction | 2 |
| 1 | direction | 4 |
| 2 | direction | 8 |
| 3 | direction | 16 |
| 4 | motor A | 2 |
| 5 | motor A | 4 |
| 6 | motor B | 2 |
| 7 | motor B | 4 |
| 8 | PWM | 2 |
| 9 | PWM | 4 |
| A | PWM | 8 |
| B | PWM | 16 |
| C | PWM | 32 |
| D | PWM | 64 |
| E | PWM | 128 |
| F | PWM | 256 |

The protocole can code 16 different directions, 2 motors and 1 PWM for the 2 motors.

Here are the possible values for direction:

| direction | Values |
| --- | --- |
| DIR_LEFT_7 | 0 |
| DIR_LEFT_6 | 1 |
| DIR_LEFT_5 | 2 |
| DIR_LEFT_4 | 3 |
| DIR_LEFT_3 | 4 |
| DIR_LEFT_2 | 5 |
| DIR_LEFT_1 | 6 |
| DIR_STRAIGHT | 7 |
| DIR_RIGHT_1 | 8 |
| DIR_RIGHT_2 | 9 |
| DIR_RIGHT_3 | A |
| DIR_RIGHT_4 | B |
| DIR_RIGHT_5 | C |
| DIR_RIGHT_6 | D |
| DIR_RIGHT_7 | E |

Please note value F is reserved for further usage.

Here are the possible values for motor:

| motor | Values |
| --- | --- |
| MOTOR_STOP | 0 |
| MOTOR_FORWARD | 1 |
| MOTOR_BACKWARD | 2 |
| MOTOR_IDLE | 3 |

PWM is a byte from 0 to 255.

## Arduino implementation

See [this file](../ArduinoControl/readme.md) for the Arduino implementation.

## Python serial port driver

Python serial port driver is using ```serial```, consider installing serial thru ```pip install serial```.

The class has few members:

* ```def __init__(self, port):```
  * used to initialize the serial port.
  * port = full name of the serial port
    * Windows: COMx where x is a number like COM3
    * Linux: /dev/ttyXYZ where XYZ is a valid tty name for example /dev/ttyS2 or /dev/ttyUSB0
  * **Important**: access to serial port under Linux requires administrator priviledges, consider launching the app in sudo mode
* ```def ChangeDirection(self, dir):```
  * change the direction of the car
  * use the direction enum
  * refer to protocol up for enum details
* ```def ChangeMotorA(self, mot):```
  * change the motor A status of the car
  * use the motor enum
  * refer to protocol up for enum details
* ```def ChangeMotorB(self, mot):```
  * change the motor B status of the car
  * use the motor enum
  * refer to protocol up for enum details
  * notes:
    * as both motors are using the same PWM, consider setting both motors with the same state. One going backward and the other one forward may damage your equipment. 
    * depending on the hbridge or equivalent you are using, the idle function may not be a valid state.
* ```def ChangePWM(self, pwm):```
  * change the speed of the car
  * use a byte from 0 to 255
* ```def ChangeAll(self, dir, motorA, motorB, pwm):```
  * change the direction, motor A, motor B and speed of the car
  * use the direction and motor enums
  * refer to protocol up for enums details

You can find an example of usage in the [SerialControlTest.py](./SerialControlTest.py) file. This is the file which has been used to run the various tests with the Arduino/ESP8266 connected and checked the expected results happened.

## **Usage**

To use the driver, you need to initialoze first the driver by passing a serial port in the initialization method:

### Initialization

```python
import SerialCommand

ser = SerialCommand.control("/dev/ttyS2")
```

Note that the ser variable needs to be initialized early in the code so you can use it in the various functions and lower in your code.

### Motors

Then you can call any of the functions. For example to move forward the motor A at 75% of the full speed:

```python
ser.ChangeMotorA(SerialCommand.motor.MOTOR_FORWARD)
# This is doing the same thing as well:
# ser.ChangeMotorA(1)
ser.ChangePWM(192)
```

Once the motor if forward, you can just change the PWM to increase or decrease the speed. So for example, changing to full  speed:

```python
ser.ChangePWM(255)
```

### Direction

Changing the direction is easy, just setup the direction you want from the enum. For example moving almost to the left:

```python
ser.ChangeDirection(SerialCommand.direction.DIR_LEFT_4)
# This is doing the same thing as well:
# ser.ChangeDirection(3)
```

Then if you want to put it back to straight:

```python
ser.ChangeDirection(SerialCommand.direction.DIR_STRAIGHT)
# This is doing the same thing as well:
# ser.ChangeDirection(7)
```

and move almost to the far right:

```python
ser.ChangeDirection(SerialCommand.direction.DIR_RIGHT_4)
# This is doing the same thing as well:
# ser.ChangeDirection(11)
```

We do recommend to use the Enums to avoid any issue but you can acheive the same using directly the int number.

## Interesting serial port usage and good things to know in Python

### bytes type doesn't exist

Python does not have per say a byte type but it has array bytes. As the protocol is base on byte, I'm using a byte array: ```self.__command = bytearray([0, 0])```

### There are Enum types of int and other ones

The concept of enum is very used in languages like C, C#, Java but most of the time in codes various classes in Python, I've seen used only variables used which are used like constants. And it is not a good way to store a data that can be changed later on. Enumls are a good way as well to pass the right value to a function if the choice is limited. 

Those are the reasons why you should definitely use Enums! And good news is that you have a variety of them in Python. In the code we've been using int enums so the value used is converted to int when needed. You can as well get the human reading part.

As an example, here is the enum for the motor states:

```python
from enum import IntEnum

class motor(IntEnum):
    MOTOR_STOP = 0
    MOTOR_FORWARD = 1
    MOTOR_BACKWARD = 2
    MOTOR_IDLE = 3
```

### bit masks are not easy to use with bytes

If you want to use bit maks on bytes as Python does not have indivisual bytes. And all need to be converted first to byte arrays. And then masks applied on the byte arrays.

The following example is used to send a change in direction to the board. As per the protocol, the lower part of the first byte is used for the direction.

```python
self.__command[0] = (self.__command[0] & 0b11110000) | (dir.to_bytes(1, byteorder='big')[0] & 0b00001111)
```

The trick is to convert first the data into a byte array, then applying the mask ```(dir.to_bytes(1, byteorder='big')[0] & 0b00001111)```. The function used to convert the ```dir``` enum to byte array is ```to_bytes```. So the enum has to be from the int byte otherwise the function doesn't work. The function exists as well in other numerical types. As we want to convert only 1 byte, we just use 1 as parameter. The parameter ```byteorder='big'``` is important as it does determine how to conver the low and high bytes first or last. So you have to know how it is coded in your platform as it is not done automatically!

Once you've converted the enum to byte array, you can finally select the first element of the array and apply the mask.

Please note that in order to clean the previous data, a reverse mask is applied first to the previous stored data to preserve the motor states and applied with an OR operator which is ```|``` in Python.

This is where you can see that Python is not a great language to manipulate low level data and use bit masks. Those operations are quite slow as well with all the conversions that apply.

## Future developments

The protocole may evolve to allow piloting both motor separately. In this case, the unsed F value for the direction will be used to tell the Ardunio that this configuration is chosen.

In this case, the PWM will be split in 2. It will allow only 16 variations instead of the 256 with the current protocol. Tests have shown that it whould be largely enough for most usage.

Other option is to add 1 more byte for PWM of motor 2.

In both cases, it will be very easy to adapt the [Arduino code](../ArduinoControl/ArduinoControl.ino).