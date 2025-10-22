#include <Wire.h>
#include <Adafruit_VL53L0X.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include <ArduinoJson.h>

// Include the streaming server functionality from app_httpd.cpp
extern void startCameraServer();
extern httpd_handle_t camera_httpd; // Declare the server handle from app_httpd.cpp

// TOF Sensor Setup
Adafruit_VL53L0X lox;
const int MEASUREMENT_INTERVAL = 200;
unsigned long lastMeasurementTime = 0;
float currentDistance = -1; // -1 = out of range

// Camera & WiFi Setup
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

const char* ssid = "tushae";
const char* password = "tusharshaw07";

// HTML Page with Auto-Refresh
const char html_page[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML>
<html>
<head>
  <title>ESP32 Camera & TOF Sensor</title>
  <meta http-equiv="refresh" content="1">
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    #distance { font-size: 24px; margin: 20px 0; }
    #stream { max-width: 640px; }
  </style>
</head>
<body>
  <h1>ESP32 Camera & TOF Sensor</h1>
  <img id="stream" src="/stream" width="640" height="480"/>
  <div id="distance">Distance: %DISTANCE% cm</div>
</body>
</html>
)rawliteral";

String getDistanceJSON() {
  StaticJsonDocument<200> doc;
  doc["distance_cm"] = currentDistance;
  doc["status"] = (currentDistance >= 0) ? "success" : "out_of_range";
  
  String jsonStr;
  serializeJson(doc, jsonStr);
  return jsonStr;
}

static esp_err_t root_handler(httpd_req_t *req) {
  String html = html_page;
  html.replace("%DISTANCE%", (currentDistance >= 0) ? String(currentDistance) : "Out of range");
  httpd_resp_send(req, html.c_str(), html.length());
  return ESP_OK;
}

static esp_err_t distance_handler(httpd_req_t *req) {
  String json = getDistanceJSON();
  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json.c_str(), json.length());
  return ESP_OK;
}

void setup() {
  Serial.begin(115200);
  
  // Initialize TOF sensor
  Wire.begin(14, 15);
  if (!lox.begin()) {
    Serial.println("Failed to boot VL53L0X!");
    while(1);
  }
  Serial.println("VL53L0X ready");

  // Initialize camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (psramFound()) {
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  s->set_framesize(s, FRAMESIZE_QVGA);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("Server started at http://");
  Serial.println(WiFi.localIP());

  // Start the camera server (includes /stream)
  startCameraServer();

  // Register additional handlers on the same server
  httpd_uri_t root_uri = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = root_handler,
    .user_ctx = NULL
  };
  httpd_register_uri_handler(camera_httpd, &root_uri);

  httpd_uri_t distance_uri = {
    .uri = "/distance",
    .method = HTTP_GET,
    .handler = distance_handler,
    .user_ctx = NULL
  };
  httpd_register_uri_handler(camera_httpd, &distance_uri);
}

void loop() {
  // Update TOF measurement
  if (millis() - lastMeasurementTime >= MEASUREMENT_INTERVAL) {
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false);

    if (measure.RangeStatus != 4) {
      currentDistance = measure.RangeMilliMeter / 10.0;
      Serial.print("Distance: ");
      Serial.print(currentDistance);
      Serial.println(" cm");
    } else {
      currentDistance = -1;
      Serial.println("Out of range (>120cm)");
    }

    lastMeasurementTime = millis();
  }
  delay(10); // Small delay to prevent tight loop
}