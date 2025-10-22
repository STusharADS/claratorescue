#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <VL53L1X.h>

VL53L1X sensor;

// ===== WiFi credentials =====
const char* ssid = "tshotspot";
const char* password = "tusharshaw08";

// ===== Web server =====
WebServer server(80);

// ===== Moving average filter =====
const int FILTER_SIZE = 5;
int readings[FILTER_SIZE];
int indexFilter = 0;
long sumFilter = 0;

// ===== Hardware pins =====
const int motorPin = 2;   
const int triggerDistance = 500; // mm → 50 cm
const int buttonPin = 13; // push button

// ===== State variables =====
int smoothDistance = 0;
bool motorState = false;
bool buttonState = false;

// ---------- WEB PAGE ----------
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>ESP32-CAM ToF Sensor</title>";
  html += "<style>body{font-family:sans-serif;text-align:center;}";
  html += "#val{font-size:32px;color:blue;}";
  html += "#motor{font-size:28px;}";
  html += "#button{font-size:28px;}</style></head><body>";
  html += "<h2>ESP32-CAM ToF Distance Data</h2>";
  html += "<p>Smoothed Distance:</p><div id='val'>Loading...</div>";
  html += "<p>Motor Status:</p><div id='motor'>Loading...</div>";
  html += "<p>Button State:</p><div id='button'>Loading...</div>";
  html += "<script>";
  html += "async function update(){";
  html += "let r = await fetch('/data');";
  html += "let j = await r.json();";
  html += "document.getElementById('val').innerText = j.distance + ' mm';";
  html += "document.getElementById('motor').innerText = j.motor;";
  html += "document.getElementById('motor').style.color = (j.motor==='ON'?'red':'green');";
  html += "document.getElementById('button').innerText = j.button;";
  html += "document.getElementById('button').style.color = (j.button==='PRESSED'?'orange':'gray');";
  html += "}";
  html += "setInterval(update,50);";  // ~20Hz updates
  html += "</script>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

// ---------- JSON API ----------
void handleData() {
  String json = "{\"distance\":" + String(smoothDistance);
  json += ",\"motor\":\"" + String(motorState ? "ON" : "OFF") + "\"";
  json += ",\"button\":\"" + String(buttonState ? "PRESSED" : "RELEASED") + "\"}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n=== VL53L1X Assistive Obstacle Detection + Web Server (Optimized) ===");

  // ESP32-CAM I2C pins
  Wire.begin(15, 14);

  if (!sensor.init()) {
    Serial.println("❌ Failed to detect and initialize VL53L1X sensor!");
    while (1);
  }

  sensor.setTimeout(500);

  // Optimized settings for 20Hz stable detection
  sensor.setDistanceMode(VL53L1X::Medium);       
  sensor.setMeasurementTimingBudget(50000);  // 50 ms integration
  sensor.startContinuous(50);                 // ~20 Hz

  sensor.setROISize(8, 16);   
  sensor.setROICenter(199);   

  // Init filter
  for (int i = 0; i < FILTER_SIZE; i++) readings[i] = 0;

  pinMode(motorPin, OUTPUT);
  digitalWrite(motorPin, LOW);

  pinMode(buttonPin, INPUT_PULLUP); // button between pin 13 and GND

  // Connect WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Connected to WiFi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Web server routes
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
}

void loop() {
  int distance = sensor.read();
  if (sensor.timeoutOccurred()) {
    Serial.println("Sensor timeout!");
    return;
  }

  // Moving average filter
  sumFilter -= readings[indexFilter];
  readings[indexFilter] = distance;
  sumFilter += distance;
  indexFilter = (indexFilter + 1) % FILTER_SIZE;
  smoothDistance = sumFilter / FILTER_SIZE;

  // Vibrate if obstacle is closer than 50 cm
  if (smoothDistance > 0 && smoothDistance < triggerDistance) {
    digitalWrite(motorPin, HIGH);
    motorState = true;
  } else {
    digitalWrite(motorPin, LOW);
    motorState = false;
  }

  // Read button (active LOW)
  buttonState = (digitalRead(buttonPin) == LOW);

  // Handle web requests
  server.handleClient();

  delay(1); // minimal delay for WiFi stability, ~20Hz loop
}
