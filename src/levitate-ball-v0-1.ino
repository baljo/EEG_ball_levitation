// Photon 2 blower: control 12 V fan via serial + WiFi TCP with repeating safety timeout
// 2025-11-15 19:45, Thomas Vikström

#include "Particle.h"

const int MOSFET_PIN = A2;                 // Grove MOSFET SIG → A2
const int PWM_MIN    = 0;
const int PWM_MAX    = 255;

const int SERVER_PORT = 9000;             // TCP port for WiFi control
const unsigned long COMMAND_TIMEOUT_MS = 30000UL;  // 30 seconds (adjust if you want)

TCPServer server(SERVER_PORT);
TCPClient client;

String serialLine;
String wifiLine;

unsigned long lastCommandMillis = 0;      // time of last valid command
bool watchdogAlreadyStopped = false;      // shared flag for watchdog

void applyPwm(int value) {
    value = constrain(value, PWM_MIN, PWM_MAX);
    analogWrite(MOSFET_PIN, value);

    Serial.printf("[CMD] PWM=%d\r\n", value);
    if (client.connected()) {
        client.printf("[CMD] PWM=%d\r\n", value);
    }

    lastCommandMillis = millis();   // update watchdog
    watchdogAlreadyStopped = false; // we just got a fresh command
}

void handleLine(const String &line) {
    if (line.length() == 0) {
        return;
    }

    int value = line.toInt();  // accepts "0".."255"
    applyPwm(value);
}

void setup() {
    pinMode(MOSFET_PIN, OUTPUT);
    analogWrite(MOSFET_PIN, 0);   // fan off at boot

    Serial.begin(115200);
    waitFor(Serial.isConnected, 5000);

    Serial.println("Photon 2 blower control (serial + WiFi TCP, with watchdog)");
    Serial.println("Send integer 0–255 + newline. WiFi port: 9000");

    // Bring up WiFi (credentials must already be configured)
    WiFi.connect();
    waitFor(WiFi.ready, 10000);

    server.begin();
    Serial.printf("WiFi ready, IP: %s, listening on TCP %d\r\n",
                  WiFi.localIP().toString().c_str(), SERVER_PORT);

    lastCommandMillis = millis(); // start watchdog from now
    watchdogAlreadyStopped = false;
}

void loop() {
    // ---- Accept new TCP client ----
    TCPClient newClient = server.available();
    if (newClient) {
        if (client.connected()) {
            client.stop();
        }
        client = newClient;
        client.println("Photon 2 blower ready. Send 0–255 + newline.");
        Serial.println("[NET] New TCP client connected");
    }

    // ---- Read from USB serial ----
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r') {
            // ignore
        } else if (c == '\n') {
            handleLine(serialLine);
            serialLine = "";
        } else {
            serialLine += c;
        }
    }

    // ---- Read from WiFi client ----
    if (client.connected()) {
        while (client.available()) {
            char c = client.read();
            if (c == '\r') {
                // ignore
            } else if (c == '\n') {
                handleLine(wifiLine);
                wifiLine = "";
            } else {
                wifiLine += c;
            }
        }
    }

    // ---- Safety timeout: stop fan after COMMAND_TIMEOUT_MS of silence ----
    unsigned long now = millis();
    if (now - lastCommandMillis > COMMAND_TIMEOUT_MS) {
        if (!watchdogAlreadyStopped) {
            analogWrite(MOSFET_PIN, 0);
            Serial.println("[SAFE] No command in timeout window → fan OFF");
            if (client.connected()) {
                client.println("[SAFE] No command in timeout window → fan OFF");
            }
            watchdogAlreadyStopped = true;
        }
    }
}
