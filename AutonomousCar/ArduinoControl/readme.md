# The Ardunio/ESP8266 code to pilot the hardware

In the [section dedicated to hardware](/electronic.md), you will find details on what is used. In short, the Arduino/ESP8266 or equivalent pilot an hbridge and a servo motor taking orders from a serial port.

The serial port protocole is [documented here](/PythonSerialControl/readme.md). And a Python driver has been developed as well to facilitate the usage. Code and description available here as well.

## Servo motor

Piloting a servo motor requires to use PWM. For the Arduino familly a servo class is provided. Just use it. Keep in mind you'll have to adjust few things.

```CPP
// Servo motor
Servo servoDirection;
#define SERVO_PIN  6   //16      // GPIO 16
// You can play with those numbers to define how left and right you want to turn
#define SERVO_LEFT  30
#define SERVO_RIGHT 145
```

You'll need of course to use a specif pin. this GPIO has to support native PWM for better performance. In case of the Arduino mini, we'll use pin 6.

Then you'll need to use a minimum value and maximum value supported by the servo motor to switch. Those are basically the minimum and maximum angle supported by the servo. You can usually find it in the documentation or just try and test. That's the pupose of the 2 define SERVO_LEFT and SERVO_RIGHT.

Then making the servo turn is straight forward base on the direction provided in the protocole which is an enum from 0 to 14.

```CPP
//Change direction
int dir = SERVO_LEFT + direction * (SERVO_RIGHT - SERVO_LEFT) / 14;
servoDirection.write(dir);
```

## Motors

Just change the pin numbers for the motor A and B. You'll have to use 2 pins. They can be any kind of digital IO.

The motor PWM need to be a hardware PWM. Be careful as they are different in every model, so refer to the documentation. PWM are not inialized the same way on Arduino and ESP8266.

If you are using an Arduino, then use this code for initialization:

```CPP
// Next lines for Arduino
pinMode(PWM_PIN, OUTPUT);
analogWrite(PWM_PIN, 0);
```

If you are using an ESP6266, use this code:

```CPP
// Next 2 lines are ESP8266 specific
analogWriteRange(255);
analogWriteFreq(19000); // 19KHz should be all ok
```

## Serial Port

Serial port is working at 115200. So a standard speed. Only 2 bytes are send, every 2 bytes the code is interpreted. The device never send anything back. You can adjust of course.