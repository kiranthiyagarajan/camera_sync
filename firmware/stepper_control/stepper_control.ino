#include <Arduino.h>
#include <math.h>

// =====================================================
// ESP32 DevKit V1 + 3 DRV8825 stepper motor drivers
// =====================================================

// GPIO assignments: do not change
const int STEP_PINS[3] = {26, 18, 32};
const int DIR_PINS[3]  = {27, 19, 33};
const int EN_PINS[3]   = {25, 23, 22};

// Motor configuration
// Assumes 200 full steps/revolution and 1/8 microstepping
const long PULSES_PER_REV = 1600;

// Each + or - command moves the selected motor by 18 degrees
const float MOVE_DEGREES = 18.0f;

// Conservative STEP pulse timing
const int STEP_HIGH_US = 5;
const int STEP_LOW_US  = 4000;

// Selected motor: -1 means none selected
int selectedMotor = -1;

// Software-estimated motor positions
float commandedDegrees[3] = {0.0f, 0.0f, 0.0f};
long pulsePosition[3] = {0, 0, 0};

// Home position tracking
long homePulsePosition[3] = {0, 0, 0};
bool homeSet[3] = {false, false, false};

// =====================================================
// Disable all motors
// DRV8825 ENABLE is active LOW
// HIGH = disabled, LOW = enabled
// =====================================================
void disableAllMotors() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(EN_PINS[i], HIGH);
  }
}

// =====================================================
// Select one motor
// =====================================================
void selectMotor(int index) {
  if (index < 0 || index > 2) {
    Serial.println("Invalid motor selection.");
    return;
  }
  disableAllMotors();
  selectedMotor = index;
  digitalWrite(EN_PINS[selectedMotor], LOW);
  Serial.print("Selected motor ");
  Serial.println(selectedMotor + 1);
  Serial.println("Motor enabled.");
  Serial.println("Press + or - to move 18 degrees.");
  Serial.println("Press 0 to set or return to home.");
}

// =====================================================
// Move the selected motor by a requested angle
// =====================================================
void moveSelectedMotor(float angleChange) {
  // Safety check: a motor must be selected first
  if (selectedMotor < 0 || selectedMotor > 2) {
    Serial.println("ERROR: Select motor 1, 2, or 3 first.");
    return;
  }
  int m = selectedMotor;

  // Calculate the new target angle
  float newAngle = commandedDegrees[m] + angleChange;

  // Convert target angle to absolute STEP pulse position
  long targetPulses = lroundf(
    newAngle * (float)PULSES_PER_REV / 360.0f
  );
  long difference = targetPulses - pulsePosition[m];
  if (difference == 0) {
    Serial.println("No STEP pulses required.");
    return;
  }

  // Set direction
  digitalWrite(DIR_PINS[m], difference > 0 ? HIGH : LOW);
  delayMicroseconds(10);
  long count = labs(difference);

  Serial.print("Motor ");
  Serial.print(m + 1);
  Serial.print(" | Target angle: ");
  Serial.print(newAngle, 1);
  Serial.print(" degrees | Pulses: ");
  Serial.println(count);

  // Generate STEP pulses
  for (long i = 0; i < count; i++) {
    digitalWrite(STEP_PINS[m], HIGH);
    delayMicroseconds(STEP_HIGH_US);
    digitalWrite(STEP_PINS[m], LOW);
    delayMicroseconds(STEP_LOW_US);
  }

  // Update software position estimate
  pulsePosition[m] = targetPulses;
  commandedDegrees[m] =
    (float)pulsePosition[m] * 360.0f / PULSES_PER_REV;
  Serial.println("Move command completed.");
}

// =====================================================
// Set home once, then return to that home position
// =====================================================
void handleHome() {
  if (selectedMotor < 0 || selectedMotor > 2) {
    Serial.println("ERROR: Select a motor first.");
    return;
  }
  int m = selectedMotor;

  // First press of 0: record the current position as home
  if (!homeSet[m]) {
    homePulsePosition[m] = pulsePosition[m];
    homeSet[m] = true;
    Serial.print("Home position recorded for motor ");
    Serial.println(m + 1);
    Serial.println(
      "This is a software reference, not a physical sensor."
    );
    return;
  }

  // Later presses of 0: return to recorded home
  long difference = homePulsePosition[m] - pulsePosition[m];
  if (difference == 0) {
    Serial.println("Motor is already at its recorded home position.");
    return;
  }

  digitalWrite(DIR_PINS[m], difference > 0 ? HIGH : LOW);
  delayMicroseconds(10);
  long count = labs(difference);

  Serial.print("Returning motor ");
  Serial.print(m + 1);
  Serial.print(" to home. STEP pulses: ");
  Serial.println(count);

  for (long i = 0; i < count; i++) {
    digitalWrite(STEP_PINS[m], HIGH);
    delayMicroseconds(STEP_HIGH_US);
    digitalWrite(STEP_PINS[m], LOW);
    delayMicroseconds(STEP_LOW_US);
  }

  pulsePosition[m] = homePulsePosition[m];
  commandedDegrees[m] =
    (float)pulsePosition[m] * 360.0f / PULSES_PER_REV;
  Serial.println("Home return command completed.");
}

// =====================================================
// Print estimated motor positions
// =====================================================
void printPositions() {
  Serial.println();
  Serial.println("=== SOFTWARE POSITION ESTIMATES ===");
  for (int i = 0; i < 3; i++) {
    Serial.print("Motor ");
    Serial.print(i + 1);
    Serial.print(" | Estimated angle: ");
    Serial.print(commandedDegrees[i], 2);
    Serial.print(" degrees");
    if (homeSet[i]) {
      float relativeAngle =
        (float)(pulsePosition[i] - homePulsePosition[i])
        * 360.0f / PULSES_PER_REV;
      Serial.print(" | Relative to home: ");
      Serial.print(relativeAngle, 2);
      Serial.print(" degrees");
    } else {
      Serial.print(" | Home not set");
    }
    Serial.println();
  }
  Serial.println("===================================");
}

// =====================================================
// Help menu
// =====================================================
void printHelp() {
  Serial.println();
  Serial.println("===== MOTOR KEYBOARD CONTROL =====");
  Serial.println("1 = Select motor 1");
  Serial.println("2 = Select motor 2");
  Serial.println("3 = Select motor 3");
  Serial.println("+ = Move selected motor +18 degrees");
  Serial.println("- = Move selected motor -18 degrees");
  Serial.println("0 = Set home / return to home");
  Serial.println("p = Print estimated positions");
  Serial.println("x = Disable all motors");
  Serial.println("? = Display this menu");
  Serial.println("==================================");
}

// =====================================================
// Setup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1500);

  for (int i = 0; i < 3; i++) {
    pinMode(STEP_PINS[i], OUTPUT);
    pinMode(DIR_PINS[i], OUTPUT);
    pinMode(EN_PINS[i], OUTPUT);
    digitalWrite(STEP_PINS[i], LOW);
    digitalWrite(DIR_PINS[i], LOW);
    // Keep all motors disabled at startup
    digitalWrite(EN_PINS[i], HIGH);
  }

  Serial.println();
  Serial.println("ESP32 three-motor controller started.");
  Serial.println("All motors are initially disabled.");
  Serial.println("Movement increment: 18 degrees.");
  Serial.println("Baud rate: 115200");
  printHelp();
}

// =====================================================
// Main loop
// =====================================================
void loop() {
  while (Serial.available() > 0) {
    char key = Serial.read();

    // Ignore line endings and spaces
    if (key == '\r' || key == '\n' || key == ' ') {
      continue;
    }

    Serial.print("Received key: ");
    Serial.println(key);

    if (key == '1') {
      selectMotor(0);
    }
    else if (key == '2') {
      selectMotor(1);
    }
    else if (key == '3') {
      selectMotor(2);
    }
    else if (key == '+' || key == '=') {
      moveSelectedMotor(MOVE_DEGREES);
    }
    else if (key == '-') {
      moveSelectedMotor(-MOVE_DEGREES);
    }
    else if (key == '0') {
      handleHome();
    }
    else if (key == 'p' || key == 'P') {
      printPositions();
    }
    else if (key == 'x' || key == 'X') {
      disableAllMotors();
      selectedMotor = -1;
      Serial.println("All motors disabled.");
      Serial.println("Select a motor before moving again.");
    }
    else if (key == '?') {
      printHelp();
    }
    else {
      Serial.println("Unknown command. Press ? for help.");
    }
  }
}
