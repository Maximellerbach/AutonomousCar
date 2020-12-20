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

// debugging board
#define BUTTON_PIN 16
#define LED_PIN 15

byte buffData[2] = {0, 0}; // one byte for the steering servo, and an other of the motor
long last_received = 0;
int maxTimout = 2000;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(200);

  // servo init
  servoSteering.attach(SERVO_PIN, SERVO_MIN, SERVO_MAX);
  servoSteering.writeMicroseconds(SERVO_NEUTRAL);

  // motor init
  motorESC.attach(ESC_PIN, ESC_MIN, ESC_MAX);
  motorESC.writeMicroseconds(ESC_NEUTRAL);

  // if the button is pressed within a second, enter calibration process
  for(int i=0; i<20; i++){ 
    delay(50);
    if(digitalRead(BUTTON_PIN)){
      blinkLED(1);
      waitButtonReleased(); // calibrate ESC
      calibrationSteps();
      break;
    }
  }
  blinkLED(1);
}

void loop() {
  if (Serial.available()) {
    Serial.readBytes(buffData, 2);
    last_received = millis();
    changeSteering();
    //changeThrottle();
  }

  // if the arduino isn't receiving anything for a given amount of time, stop the motor and servo
  //else if (abs(millis()-last_received)>maxTimout){
  //  servoSteering.writeMicroseconds(SERVO_NEUTRAL);
  //  motorESC.writeMicroseconds(ESC_NEUTRAL);
  //}
}

void changeSteering() {
  int decoded_steering = buffData[0]; // cast byte to int
  
  int steering = SERVO_MIN + decoded_steering/255 * (SERVO_MAX - SERVO_MIN);
  servoSteering.writeMicroseconds(steering);
}

void changeThrottle() {
  int decoded_trottle = buffData[1]; // cast byte to int
  
  int throttle = ESC_MIN + decoded_trottle/255 * (ESC_MAX - ESC_MIN);
  motorESC.writeMicroseconds(throttle);
}

void calibrationSteps(){
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
  motorESC.writeMicroseconds((ESC_MIN+ESC_MAX)/2);

}

void waitButtonClicked(){
  waitButtonPressed();
  waitButtonReleased();
}

void waitButtonPressed(){
  while(!digitalRead(BUTTON_PIN)){ // wait for button to be pressed
    //Serial.println(digitalRead(BUTTON_PIN)); // for debugging purpose
    delay(50);
  }
}

void waitButtonReleased(){
  while(digitalRead(BUTTON_PIN)){ // wait for button to be released
    delay(50);
  }
}

void blinkLED(int rep){
  for(int i=0; i<rep; i++){
    digitalWrite(LED_PIN, HIGH);
    delay(250);
    digitalWrite(LED_PIN, LOW);
    delay(250);
  }
}
