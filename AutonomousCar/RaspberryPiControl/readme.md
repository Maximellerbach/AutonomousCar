# Simple classes to pilot motors, servo and PWM from Python classes

This part of the code has been used in early tests during choices of architecture. It is using wiringpi. So don't forget to install the class before using it.

Those classes can be used very easilly with a Flask server like in the [other part of the project](../PythonSerialControl/webcontrol.md). You'll just have to call those classes instead of the serial port. This will allow to drive your car from a Raspberry Pi using a web interface :-)

## Motor

The motro class allow to pilot motors which can be associated with a PWM or not. Please open the source code and check the comments, they contains important information if you want to reuse this code.

One of the most important point is that Raspberry only have few native PWM, so make sure you are using the right ones. Otherwise, any other pin can be used to pilot the hbridge.

## Servo motor

Servo motor is using as well PWM. So please make sure you'll be using the right pins.

## Test

the project contains as well test files which will allow you to check if the expected behavior is the correct one.