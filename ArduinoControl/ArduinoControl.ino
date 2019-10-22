#include <Servo.h> 

byte buffData[2] = {0, 0};

typedef enum Direction {
  DIR_LEFT_7 = 0,
  DIR_LEFT_6 = 1,
  DIR_LEFT_5 = 2,
  DIR_LEFT_4 = 3,
  DIR_LEFT_3 = 4,
  DIR_LEFT_2 = 5,
  DIR_LEFT_1 = 6,
  DIR_STRAIGHT = 7,
  DIR_RIGHT_1 = 8,
  DIR_RIGHT_2 = 9,
  DIR_RIGHT_3 = 10,
  DIR_RIGHT_4 = 11,
  DIR_RIGHT_5 = 12,
  DIR_RIGHT_6 = 13,
  DIR_RIGHT_7 = 14
};

typedef enum Motor {
  MOTOR_STOP = 0,
  MOTOR_FORWARD = 1,
  MOTOR_BACKWARD = 2,
  MOTOR_IDLE = 3
};

typedef enum PWMType {
  PWM_BOTH = 0,
  PWM_INDEPENDANT = 1
};

int PWM;
Direction direction;
Motor motorA;
Motor motorB;
PWMType PWMtype;

// Servo motor
Servo servoDirection;
#define SERVO_PIN  6   //16      // GPIO 16
// You can play with those numbers to define how left and right you want to turn
#define SERVO_LEFT  30
#define SERVO_RIGHT 145

// this is needed for every servo motor
// every servo has specific timing
//#define SERVO_MIN 900
//#define SERVO_MAX 1900

// Motor A
#define MOTOR_A_PIN_A 8 // 5
#define MOTOR_A_PIN_B 7 //4
// Motor B
#define MOTOR_B_PIN_A 2 //0
#define MOTOR_B_PIN_B 4 //2
//PWM pin
#define PWM_PIN 3 //14

void setup() {
  // put your setup code here, to run once:
  // start serial port
  Serial.begin(115200);
  Serial.setTimeout(200);
  //Serial.println("initialized");
  direction = DIR_STRAIGHT;
  PWMtype = PWM_BOTH;
  motorA = MOTOR_IDLE;
  motorB = MOTOR_IDLE;
  // Setting up the Servo
  //servoDirection.attach(SERVO_PIN, SERVO_MIN, SERVO_MAX); 
  servoDirection.attach(SERVO_PIN);
  // Setting up the motors
  pinMode(MOTOR_A_PIN_A, OUTPUT);
  pinMode(MOTOR_A_PIN_B, OUTPUT);
  pinMode(MOTOR_B_PIN_A, OUTPUT);
  pinMode(MOTOR_B_PIN_B, OUTPUT);
  digitalWrite(MOTOR_A_PIN_A, LOW);
  digitalWrite(MOTOR_A_PIN_B, LOW);
  digitalWrite(MOTOR_B_PIN_A, LOW);
  digitalWrite(MOTOR_B_PIN_B, LOW);
  // PWM
  //analogWriteResolution(8); // Setup the resolution for 8 bits in Arduino
  // Next 2 lines are ESP8266 specific
  //analogWriteRange(255);
  //analogWriteFreq(19000); // 19KHz should be all ok
  // Next lines for Arduino
  pinMode(PWM_PIN, OUTPUT);
  analogWrite(PWM_PIN, 0);
}

void loop() {

  if (Serial.available()) {
    Serial.readBytes(buffData, 2);
    DecryptSerial();
  }
}

void DecryptSerial()
{
  direction = (Direction)(buffData[0] & 0b1111);
  motorA = (Motor)((buffData[0] >> 4) & 0b11);
  motorB = (Motor)((buffData[0] >> 6) & 0b11);
  PWM = buffData[1];
  //debug
  // Serial.print("Direction: ");
  // Serial.print(direction);
  // Serial.print(" MotorA: ");
  // Serial.print(motorA);
  // Serial.print(" MotorB: ");
  // Serial.print(motorB);
  // Serial.print(" PWM: ");
  // Serial.println(PWM);
  //end debug

  //Change direction
  int dir = SERVO_LEFT + direction * (SERVO_RIGHT - SERVO_LEFT) / 14;
  servoDirection.write(dir);

  // Motor A and Motor B
  if (motorA == MOTOR_IDLE)
  {
    digitalWrite(MOTOR_A_PIN_A, LOW);
    digitalWrite(MOTOR_A_PIN_B, LOW);
  } else if (motorA == MOTOR_STOP)
  {
    digitalWrite(MOTOR_A_PIN_A, HIGH);
    digitalWrite(MOTOR_A_PIN_B, HIGH);
  } else if (motorA == MOTOR_FORWARD)
  {
    digitalWrite(MOTOR_A_PIN_A, HIGH);
    digitalWrite(MOTOR_A_PIN_B, LOW);
  } else if (motorA == MOTOR_BACKWARD)
  {
    digitalWrite(MOTOR_A_PIN_A, LOW);
    digitalWrite(MOTOR_A_PIN_B, HIGH);
  }
  if (motorB == MOTOR_IDLE)
  {
    digitalWrite(MOTOR_B_PIN_A, LOW);
    digitalWrite(MOTOR_B_PIN_B, LOW);
  } else if (motorB == MOTOR_STOP)
  {
    digitalWrite(MOTOR_B_PIN_A, HIGH);
    digitalWrite(MOTOR_B_PIN_B, HIGH);
  } else if (motorB == MOTOR_FORWARD)
  {
    digitalWrite(MOTOR_B_PIN_A, HIGH);
    digitalWrite(MOTOR_B_PIN_B, LOW);
  } else if (motorB == MOTOR_BACKWARD)
  {
    digitalWrite(MOTOR_B_PIN_A, LOW);
    digitalWrite(MOTOR_B_PIN_B, HIGH);
  }

  //PWM
  analogWrite(PWM_PIN, PWM);
}
