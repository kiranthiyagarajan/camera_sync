#include <Arduino.h>
#include <Preferences.h>
#include <math.h>

// =====================================================
// ESP32 DevKit V1 + 3 DRV8825 stepper motor drivers  (v2)
//
// All motors stay enabled (holding) so tendon tension can't back-drive them.
// Positions are counted relative to home and saved to flash after every move,
// so they survive a reset or power cycle.
//
// Line commands (end with newline):
//   G a1 a2 a3   move all motors together to these angles (degrees from home)
//   H            set the current position of all motors as home
//   P            print "POS a1 a2 a3 home=0|1 enabled=0|1"
// Single keys (as in v1):
//   1/2/3 select a motor, + / - move it 18 degrees, 0 set or return to home,
//   p print positions, x disable all motors (e enables them again), ? help
// A move started with G or + / - stops early if 'x' arrives while it runs.
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

// STEP pulse timing (same rate as v1: one pulse per 4 ms on the fastest motor)
const int STEP_HIGH_US = 5;
const unsigned long STEP_PERIOD_US = 4005;

int selectedMotor = -1;
long pulsePosition[3] = {0, 0, 0};  // relative to home
bool homeSet = false;
bool enabled = false;

Preferences prefs;

float toDegrees(long pulses) { return (float)pulses * 360.0f / PULSES_PER_REV; }
long toPulses(float degrees) { return lroundf(degrees * (float)PULSES_PER_REV / 360.0f); }

void savePositions() {
  prefs.putBytes("pos", pulsePosition, sizeof(pulsePosition));
  prefs.putBool("home", homeSet);
}

// DRV8825 ENABLE is active LOW
void setEnabled(bool on) {
  for (int i = 0; i < 3; i++) {
    digitalWrite(EN_PINS[i], on ? LOW : HIGH);
  }
  enabled = on;
}

// Move all motors to target pulse positions together (straight line in motor space).
// Returns false if stopped early by 'x'.
bool moveTo(const long target[3]) {
  long delta[3], count[3], done[3] = {0, 0, 0};
  long longest = 0;
  for (int i = 0; i < 3; i++) {
    delta[i] = target[i] - pulsePosition[i];
    count[i] = labs(delta[i]);
    digitalWrite(DIR_PINS[i], delta[i] > 0 ? HIGH : LOW);
    if (count[i] > longest) longest = count[i];
  }
  if (longest == 0) return true;
  if (!enabled) setEnabled(true);
  delayMicroseconds(10);

  unsigned long next = micros();
  for (long k = 1; k <= longest; k++) {
    while ((long)(micros() - next) < 0) {}
    next += STEP_PERIOD_US;
    for (int i = 0; i < 3; i++) {
      // Spread each motor's pulses evenly over the move
      long due = (count[i] * k) / longest;
      if (due > done[i]) {
        digitalWrite(STEP_PINS[i], HIGH);
      }
    }
    delayMicroseconds(STEP_HIGH_US);
    for (int i = 0; i < 3; i++) {
      long due = (count[i] * k) / longest;
      if (due > done[i]) {
        digitalWrite(STEP_PINS[i], LOW);
        done[i] = due;
        pulsePosition[i] += delta[i] > 0 ? 1 : -1;
      }
    }
    if (Serial.available() > 0 && (Serial.peek() == 'x' || Serial.peek() == 'X')) {
      savePositions();
      Serial.println("STOPPED");
      return false;
    }
  }
  savePositions();
  return true;
}

void printPos() {
  Serial.print("POS");
  for (int i = 0; i < 3; i++) {
    Serial.print(" ");
    Serial.print(toDegrees(pulsePosition[i]), 3);
  }
  Serial.print(" home=");
  Serial.print(homeSet ? 1 : 0);
  Serial.print(" enabled=");
  Serial.println(enabled ? 1 : 0);
}

void setHome() {
  for (int i = 0; i < 3; i++) pulsePosition[i] = 0;
  homeSet = true;
  savePositions();
}

// G a1 a2 a3: absolute angles from home
void handleGoto(String args) {
  float a[3];
  if (sscanf(args.c_str(), "%f %f %f", &a[0], &a[1], &a[2]) != 3) {
    Serial.println("ERR usage: G a1 a2 a3");
    return;
  }
  if (!homeSet) {
    Serial.println("ERR home not set (send H)");
    return;
  }
  long target[3];
  for (int i = 0; i < 3; i++) target[i] = toPulses(a[i]);
  bool ok = moveTo(target);
  Serial.print(ok ? "OK " : "STOPPED ");
  printPos();
}

void moveSelected(float angleChange) {
  if (selectedMotor < 0) {
    Serial.println("ERROR: Select motor 1, 2, or 3 first.");
    return;
  }
  long target[3] = {pulsePosition[0], pulsePosition[1], pulsePosition[2]};
  target[selectedMotor] += toPulses(angleChange);
  Serial.print("Motor ");
  Serial.print(selectedMotor + 1);
  Serial.print(" | Target angle: ");
  Serial.print(toDegrees(target[selectedMotor]), 1);
  Serial.println(" degrees");
  if (moveTo(target)) Serial.println("Move command completed.");
}

// First press of 0 sets home (for all motors), later presses return the selected motor home
void handleHome() {
  if (!homeSet) {
    setHome();
    Serial.println("Home position recorded for all motors.");
    return;
  }
  if (selectedMotor < 0) {
    Serial.println("ERROR: Select a motor first.");
    return;
  }
  long target[3] = {pulsePosition[0], pulsePosition[1], pulsePosition[2]};
  target[selectedMotor] = 0;
  if (moveTo(target)) Serial.println("Home return command completed.");
}

void printPositions() {
  Serial.println();
  Serial.println("=== SOFTWARE POSITION ESTIMATES ===");
  for (int i = 0; i < 3; i++) {
    Serial.print("Motor ");
    Serial.print(i + 1);
    Serial.print(" | Relative to home: ");
    Serial.print(toDegrees(pulsePosition[i]), 2);
    Serial.println(" degrees");
  }
  Serial.println(homeSet ? "Home set." : "Home not set.");
  Serial.println(enabled ? "Motors enabled (holding)." : "Motors disabled.");
  Serial.println("===================================");
}

void printHelp() {
  Serial.println();
  Serial.println("===== MOTOR CONTROL (v2) =====");
  Serial.println("G a1 a2 a3 = Move all motors to angles from home (degrees)");
  Serial.println("H = Set current position as home");
  Serial.println("P = Print positions (one line)");
  Serial.println("1/2/3 = Select motor, + / - = move it 18 degrees");
  Serial.println("0 = Set home / return selected motor to home");
  Serial.println("p = Print estimated positions");
  Serial.println("x = Disable all motors, e = enable all motors");
  Serial.println("? = Display this menu");
  Serial.println("==============================");
}

void setup() {
  for (int i = 0; i < 3; i++) {
    pinMode(STEP_PINS[i], OUTPUT);
    pinMode(DIR_PINS[i], OUTPUT);
    pinMode(EN_PINS[i], OUTPUT);
    digitalWrite(STEP_PINS[i], LOW);
    digitalWrite(DIR_PINS[i], LOW);
  }
  // Hold the motors straight away so the robot doesn't relax during boot
  setEnabled(true);

  prefs.begin("stepper", false);
  if (prefs.getBytesLength("pos") == sizeof(pulsePosition)) {
    prefs.getBytes("pos", pulsePosition, sizeof(pulsePosition));
  }
  homeSet = prefs.getBool("home", false);

  Serial.begin(115200);
  Serial.setTimeout(200);
  Serial.println();
  Serial.println("ESP32 three-motor controller v2 started.");
  printHelp();
  printPos();
}

void loop() {
  while (Serial.available() > 0) {
    char key = Serial.read();
    if (key == '\r' || key == '\n' || key == ' ') {
      continue;
    }
    if (key == 'G') {
      handleGoto(Serial.readStringUntil('\n'));
      continue;
    }

    Serial.print("Received key: ");
    Serial.println(key);

    if (key == '1' || key == '2' || key == '3') {
      selectedMotor = key - '1';
      Serial.print("Selected motor ");
      Serial.println(selectedMotor + 1);
    }
    else if (key == '+' || key == '=') moveSelected(MOVE_DEGREES);
    else if (key == '-') moveSelected(-MOVE_DEGREES);
    else if (key == '0') handleHome();
    else if (key == 'H') { setHome(); Serial.println("Home set to current position."); }
    else if (key == 'P') printPos();
    else if (key == 'p') printPositions();
    else if (key == 'x' || key == 'X') {
      setEnabled(false);
      Serial.println("All motors disabled.");
    }
    else if (key == 'e') {
      setEnabled(true);
      Serial.println("All motors enabled.");
    }
    else if (key == '?') printHelp();
    else Serial.println("Unknown command. Press ? for help.");
  }
}
