#include <Servo.h>

// Servo
#define SERVO_PIN 6
#define SERVO_MIN 900
#define SERVO_MAX 2100
#define SERVO_NEUTRAL 1500
#define SERVO_DEADBAND 2
Servo servoSteering;

// Motor
#define ESC_PIN 5
#define ESC_MIN 1000
#define ESC_MAX 2000
#define ESC_NEUTRAL 1500
Servo motorESC;

// Sensor
#define SENSOR_PIN 3

// debugging board
#define BUTTON_PIN 16
#define LED_PIN 15

// variables used to read serial
byte dummyBuff[1] = {0};
byte buffData[4] = {0, 0, 0, 0}; // one byte for start, one byte for the steering servo, an other of the motor and one byte for end
int expected_start = 255;
int expected_end = 0;

// variables to detect last received serial message (safety)
long last_received = 0;
int maxTimout = 500;

// variables to read PWM pulses
unsigned long timer_start = 0;
unsigned long last_interrupt_time = 0;
int motor_speed = 0;
int prev_motor_speed = 0;

void setup()
{
  Serial.begin(115200);
  Serial.setTimeout(200);

  // servo init
  servoSteering.attach(SERVO_PIN, SERVO_MIN, SERVO_MAX);
  servoSteering.writeMicroseconds(SERVO_NEUTRAL);

  // motor init
  motorESC.attach(ESC_PIN, ESC_MIN, ESC_MAX);
  motorESC.writeMicroseconds(ESC_NEUTRAL);

  // sensor init
  pinMode(SENSOR_PIN, INPUT);
  // attachInterrupt(SENSOR_PIN, signalChange, CHANGE);

  // if the button is pressed within a second, enter calibration process
  for (int i = 0; i < 20; i++)
  {
    delay(50);
    if (digitalRead(BUTTON_PIN))
    {
      blinkLED(1);
      waitButtonReleased(); // calibrate ESC
      calibrationSteps();
      break;
    }
  }
  blinkLED(1);
}

void loop()
{
  if (Serial.available())
  {

    // write rpm sensor data to the serial
    //if (motor_speed != prev_motor_speed) 
    //{
    Serial.println(motor_speed);
    //prev_motor_speed = motor_speed;
    //}

    // read the data from the serial
    Serial.readBytes(buffData, 4);

    if (buffData[0] == expected_start && buffData[3] == expected_end) // check wether we are reading the right data buffer
    {
      last_received = millis();
      changeSteering();
      changeThrottle();
    }
    else
    {
      Serial.readBytes(dummyBuff, 1); // needed to adjust start and end of the message
    }
  }

  // if the arduino isn't receiving anything for a given amount of time, stop the motor and servo
  else if (!Serial.available() || millis() - last_received > maxTimout)
  {
    servoSteering.writeMicroseconds(SERVO_NEUTRAL);
    motorESC.writeMicroseconds(ESC_NEUTRAL);
  }
}

void changeSteering()
{
  float decoded_steering = buffData[1]; // cast byte to int

  int steering = SERVO_MAX - decoded_steering / 255 * (SERVO_MAX - SERVO_MIN);
  servoSteering.writeMicroseconds(steering);
}

void changeThrottle()
{
  float decoded_trottle = buffData[2]; // cast byte to int

  int throttle = ESC_MIN + decoded_trottle / 255 * (ESC_MAX - ESC_MIN);
  motorESC.writeMicroseconds(throttle);
}

void signalChange() // this function will be called on state change of SENSOR_PIN
{
  last_interrupt_time = micros();
  if (digitalRead(SENSOR_PIN) == HIGH)
  {
    timer_start = last_interrupt_time;
  }
  else
  {
    if (timer_start != 0)
    {
      int p_time = last_interrupt_time - timer_start;
      timer_start = 0;
      
      // check that there is no overflow 
      if (p_time > 0) {motor_speed = p_time;}
    }
  }
}

void calibrationSteps()
{
  // neutral pwm
  waitButtonClicked();
  blinkLED(1);
  motorESC.writeMicroseconds(ESC_NEUTRAL);

  // max pwm
  waitButtonClicked();
  blinkLED(2);
  motorESC.writeMicroseconds(ESC_MAX);

  // min pwm
  waitButtonClicked();
  blinkLED(3);
  motorESC.writeMicroseconds(ESC_MIN);

  delay(2900); // wait less than 3 seconds and set to neutral point to avoid motor to start at full power
  motorESC.writeMicroseconds((ESC_MIN + ESC_MAX) / 2);
}

void waitButtonClicked()
{
  waitButtonPressed();
  waitButtonReleased();
}

void waitButtonPressed()
{
  while (!digitalRead(BUTTON_PIN))
  { // wait for button to be pressed
    // Serial.println(digitalRead(BUTTON_PIN)); // for debugging purpose
    delay(50);
  }
}

void waitButtonReleased()
{
  while (digitalRead(BUTTON_PIN))
  { // wait for button to be released
    delay(50);
  }
}

void blinkLED(int rep)
{
  for (int i = 0; i < rep; i++)
  {
    digitalWrite(LED_PIN, HIGH);
    delay(250);
    digitalWrite(LED_PIN, LOW);
    delay(250);
  }
}
