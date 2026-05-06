#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>
#include <math.h>
#include <string.h>

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_SDA 4
#define OLED_SCL 5

// MAX98357A I2S - must not overlap with OLED (4/5)
#define AUDIO_BCK 10
#define AUDIO_WS 11
#define AUDIO_DATA 12
#define I2S_PORT I2S_NUM_0
#define AUDIO_GAIN 1.0f
#define AUDIO_NOISE_GATE 0

// Push button: connect GPIO13 to GND. Uses internal pull-up.
#define CHAT_BUTTON_PIN 13
#define CHAT_BUTTON_DEBOUNCE_MS 60

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// =======================
// WIFI & API
// =======================
const char *ssid = "Liz";
const char *password = "calvaria648";

const char *apiHost = "acorn-straw-tux.ngrok-free.dev";
const int apiPort = 443;
const char *apiPath = "/api/v1/chat";

const char *quickChats[] = {
    "liz, get angry please",
    "liz, i'm happy right now",
    "liz, i'm sad",
    "liz, i'm excited"};
const int quickChatCount = sizeof(quickChats) / sizeof(quickChats[0]);
int quickChatIndex = 0;

// =======================
// EMOTION
// =======================
enum Emotion
{
  SAD,
  HAPPY,
  STRESSED,
  SQUINT_HAPPY,
  SHOCKED
};

Emotion currentEmotion = HAPPY;

unsigned long lastTime = 0;
float t = 0.0;

const int transitionSteps = 28;
const int transitionDelay = 12;

// =======================
// FUNCTION DECLARATION
// =======================
void sendChatToLiz(String message);
void changeFaceFromApi(String rawEmotion);
void drawEmotion(Emotion e, float time, int maskAmount, bool revealMode);
void fastPixelTransition(Emotion from, Emotion to);
float easeOutCubic(float x);
uint8_t hashPixel(int x, int y);
bool allowPixel(int x, int y, int amount, bool revealMode);
void mpixel(int x, int y, int amount, bool revealMode);
void mline(int x0, int y0, int x1, int y1, int amount, bool revealMode);
void mrect(int x, int y, int w, int h, int amount, bool revealMode);
void mfillCircle(int cx, int cy, int r, int amount, bool revealMode);
void mcircle(int cx, int cy, int r, int amount, bool revealMode);
void marcLeft(int cx, int cy, int r, int amount, bool revealMode);
void marcRight(int cx, int cy, int r, int amount, bool revealMode);
void drawParentheses(int ox, int oy, int amount, bool revealMode);
void drawBlush(int x, int y, int amount, bool revealMode);
void drawSadFace(float time, int amount, bool revealMode);
void drawSadHands(float time, int ox, int oy, int amount, bool revealMode);
void drawCryingEye(int x, int y, float tearProgress, int amount, bool revealMode);
void drawWavySadMouth(int cx, int cy, float time, int amount, bool revealMode);
void drawHappyFace(float time, int amount, bool revealMode);
void drawHappyEyeSmooth(int x, int y, bool closed, int amount, bool revealMode);
void drawHappyMouthSmooth(int cx, int cy, float open, int amount, bool revealMode);
void drawStressedFace(float time, int amount, bool revealMode);
void drawGreaterEyeSmooth(int x, int y, int press, int amount, bool revealMode);
void drawLessEyeSmooth(int x, int y, int press, int amount, bool revealMode);
void drawStressedMouthSmooth(int cx, int cy, float time, int amount, bool revealMode);
void drawSquintHappyFace(float time, int amount, bool revealMode);
void drawSquintEye(int x, int y, int lift, int amount, bool revealMode);
void drawWideSmile(int cx, int cy, float open, int amount, bool revealMode);
void drawShockedFace(float time, int amount, bool revealMode);
void drawSideLookingCircleEye(int x, int y, int r, int lookDir, int amount, bool revealMode);
void drawSquareMouth(int cx, int cy, int size, int amount, bool revealMode);
void drawSoftBlush(int x, int y, float time, int amount, bool revealMode);
void drawExclamation(int x, int y, int amount, bool revealMode);
bool i2sInit(int sampleRate);
void i2sDeinit();
void i2sWriteSample(int16_t sample);
void streamPlayAudio(WiFiClientSecure &client, unsigned long deadline);

// =======================
// SETUP
// =======================
void setup()
{
  Serial.begin(115200);
  pinMode(CHAT_BUTTON_PIN, INPUT_PULLUP);

  Wire.begin(OLED_SDA, OLED_SCL);
  // Keep Adafruit_SSD1306 from calling Wire.begin() again and losing
  // the ESP32-S3 custom SDA/SCL pins configured above.
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C, true, false)) {
    Serial.println("OLED init failed. Check SDA/SCL pins and OLED address.");
  }
  display.clearDisplay();
  display.display();

  // Start WiFi in background — animation runs immediately in loop()
  WiFi.begin(ssid, password);
  Serial.println("WiFi connecting...");

  lastTime = millis();
}

// =======================
// LOOP
// =======================
void loop()
{
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0;
  lastTime = now;
  t += dt;

  display.clearDisplay();
  drawEmotion(currentEmotion, t, 255, true);
  display.display();

  // Print once when WiFi first connects
  static bool wifiReady = false;
  bool wifiNow = (WiFi.status() == WL_CONNECTED);
  if (wifiNow && !wifiReady) {
    Serial.println("WiFi connected! IP: " + WiFi.localIP().toString());
    Serial.println("Ketik di Serial Monitor...");
  }
  wifiReady = wifiNow;

  static int lastButtonReading = HIGH;
  static int stableButtonState = HIGH;
  static unsigned long lastButtonChange = 0;

  int buttonReading = digitalRead(CHAT_BUTTON_PIN);
  if (buttonReading != lastButtonReading) {
    lastButtonChange = now;
    lastButtonReading = buttonReading;
  }

  if ((now - lastButtonChange) > CHAT_BUTTON_DEBOUNCE_MS && buttonReading != stableButtonState) {
    stableButtonState = buttonReading;
    if (stableButtonState == LOW) {
      if (!wifiReady) {
        Serial.println("WiFi belum terhubung, tombol diabaikan dulu...");
      }
      else {
        const char *message = quickChats[quickChatIndex];
        Serial.printf("Button chat %d/%d: %s\n", quickChatIndex + 1, quickChatCount, message);
        quickChatIndex = (quickChatIndex + 1) % quickChatCount;
        sendChatToLiz(String(message));
      }
    }
  }

  if (Serial.available())
  {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0)
    {
      if (!wifiReady)
        Serial.println("WiFi belum terhubung, tunggu sebentar...");
      else
      {
        Serial.println("Chatmu: " + input);
        sendChatToLiz(input);
      }
    }
  }

  delay(22);
}

// =======================
// API CALL (HTTPS — ESP32 mbedTLS handles ngrok TLS fine)
// =======================
void sendChatToLiz(String message)
{
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi tidak terhubung, animasi tetap jalan.");
    return;
  }

  // Transition to idle HAPPY before connecting
  if (currentEmotion != HAPPY)
    fastPixelTransition(currentEmotion, HAPPY);
  currentEmotion = HAPPY;

  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(8); // fail fast if server is down — keeps LCD responsive

  Serial.print("Connecting to Liz (HTTPS)... ");
  if (!client.connect(apiHost, apiPort))
  {
    Serial.println("FAILED");
    return;
  }
  Serial.println("OK");

  StaticJsonDocument<128> req;
  req["message"] = message;
  String body;
  serializeJson(req, body);

  // HTTP/1.0 — avoids chunked transfer encoding
  client.printf("POST %s HTTP/1.0\r\n", apiPath);
  client.printf("Host: %s\r\n", apiHost);
  client.print("Content-Type: application/json\r\n");
  client.print("ngrok-skip-browser-warning: true\r\n");
  client.printf("Content-Length: %d\r\n", body.length());
  client.print("\r\n");
  client.print(body);

  // Wait for response — keep animation alive
  unsigned long waitStart = millis();
  while (!client.available())
  {
    if (!client.connected())
    {
      Serial.println("Server menutup koneksi tanpa respons");
      client.stop();
      return;
    }
    if (millis() - waitStart > 60000)
    {
      Serial.println("Timeout: server tidak merespons dalam 60s");
      client.stop();
      return;
    }
    unsigned long now = millis();
    t += (now - lastTime) / 1000.0;
    lastTime = now;
    display.clearDisplay();
    drawEmotion(currentEmotion, t, 255, true);
    display.display();
    delay(22);
  }
  Serial.printf("Response diterima setelah %lums\n", millis() - waitStart);

  // Read status line
  String statusLine = client.readStringUntil('\n');
  Serial.println("Status: " + statusLine);
  int httpCode = (statusLine.indexOf("200") >= 0) ? 200 : 0;

  // Read headers, find Content-Length
  int contentLength = -1;
  while (client.available() || client.connected())
  {
    String line = client.readStringUntil('\n');
    line.trim();
    if (line.length() == 0)
      break;
    if (line.startsWith("Content-Length:"))
      contentLength = line.substring(15).toInt();
  }

  if (httpCode == 200)
  {
    // Stream the body without buffering — the audio_payload alone can be 200-300 KB,
    // far too large to hold in a String on the ESP32-S3 heap.
    unsigned long deadline = millis() + 120000;
    int bodyBytesRead = 0;

    // nextCh: read one byte from the HTTPS stream, blocking until data arrives
    auto nextCh = [&]() -> int {
      if (contentLength >= 0 && bodyBytesRead >= contentLength) return -3;
      while (millis() < deadline) {
        if (client.available()) {
          bodyBytesRead++;
          return (uint8_t)client.read();
        }
        if (!client.connected()) return -2;
        delay(1);
      }
      return -1;
    };

    // scanTo: advance stream until the literal pattern is matched
    auto scanTo = [&](const char *pat) -> bool {
      int plen = strlen(pat), m = 0;
      while (true) {
        int c = nextCh();
        if (c < 0) return false;
        if ((char)c == pat[m]) { if (++m == plen) return true; }
        else m = ((char)c == pat[0]) ? 1 : 0;
      }
    };

    // readQuoted: read a JSON string value (opening " already consumed)
    auto readQuoted = [&]() -> String {
      String s;
      while (true) {
        int c = nextCh();
        if (c < 0 || c == '"') break;
        if (c == '\\') { nextCh(); continue; }
        s += (char)c;
      }
      return s;
    };

    auto nextNonSpace = [&]() -> int {
      int c;
      do {
        c = nextCh();
        if (c < 0) return c;
      } while (c == ' ' || c == '\r' || c == '\n' || c == '\t');

      return c;
    };

    String textVal = "";
    String emotionVal = "";
    bool audioPlayed = false;

    auto applyEmotion = [&](String value) {
      emotionVal = value;
      Serial.println("Emotion: " + emotionVal);
      changeFaceFromApi(emotionVal);
    };

    // Parse metadata: {"emotion":"ANGRY"} without buffering the whole response.
    // Opening { has already been consumed.
    auto readMetadataObject = [&]() -> bool {
      int depth = 1;
      while (depth > 0) {
        int c = nextCh();
        if (c < 0) return false;

        if (c == '"') {
          String metaKey = readQuoted();
          int sep = nextNonSpace();
          if (sep != ':') continue;

          int metaValueStart = nextNonSpace();
          if (metaKey == "emotion" && metaValueStart == '"') {
            applyEmotion(readQuoted());
            return true;
          }

          if (metaValueStart == '"') {
            readQuoted();
          }
          else if (metaValueStart == '{') {
            depth++;
          }
          else if (metaValueStart == '}') {
            depth--;
          }
        }
        else if (c == '{') {
          depth++;
        }
        else if (c == '}') {
          depth--;
        }
      }

      return false;
    };

    while (scanTo("\"")) {
      String key = readQuoted();
      int sep = nextNonSpace();
      if (sep != ':') continue;

      int valueStart = nextNonSpace();
      if (key == "metadata" && valueStart == '{') {
        bool foundEmotion = readMetadataObject();
        if (audioPlayed && foundEmotion)
          break;
        continue;
      }

      if (valueStart != '"') continue;

      if (key == "audio_payload" || key == "audio_base64") {
        Serial.println("Audio payload found, streaming to I2S...");
        streamPlayAudio(client, deadline);
        audioPlayed = true;
        if (emotionVal.length() > 0)
          break;

        // If the API sends emotion after the large audio field, keep scanning
        // briefly after playback without risking a long post-audio stall.
        deadline = millis() + 3000;
        continue;
      }
      else {
        String value = readQuoted();
        if (key == "text") {
          textVal = value;
        }
        else if (key == "emotion") {
          applyEmotion(value);
          if (audioPlayed)
            break;
        }
      }
    }

    if (!audioPlayed)
      Serial.println("No audio_payload/audio_base64 found in response.");
    Serial.println("Berhasil!");
  }
  else
  {
    Serial.printf("Gagal, HTTP code: %d\n", httpCode);
  }

  client.stop();
}

// =======================
// EMOTION SWITCH
// =======================
void changeFaceFromApi(String rawEmotion)
{
  rawEmotion.trim();
  rawEmotion.toUpperCase();

  Emotion newEmotion = HAPPY;
  if (rawEmotion == "SAD")
    newEmotion = SAD;
  else if (rawEmotion == "HAPPY")
    newEmotion = HAPPY;
  else if (rawEmotion == "ANGRY" || rawEmotion == "STRESSED")
    newEmotion = STRESSED;
  else if (rawEmotion == "EXCITED" || rawEmotion == "SQUINT_HAPPY")
    newEmotion = SQUINT_HAPPY;
  else if (rawEmotion == "SHOCKED" || rawEmotion == "SURPRISED")
    newEmotion = SHOCKED;
  else
    Serial.println("Emotion tidak dikenal, fallback ke HAPPY: " + rawEmotion);

  if (newEmotion != currentEmotion)
    fastPixelTransition(currentEmotion, newEmotion);

  currentEmotion = newEmotion;
}

// =======================
// AUDIO (MAX98357A I2S)
// =======================
static int b64val(char c)
{
  if (c >= 'A' && c <= 'Z')
    return c - 'A';
  if (c >= 'a' && c <= 'z')
    return c - 'a' + 26;
  if (c >= '0' && c <= '9')
    return c - '0' + 52;
  if (c == '+')
    return 62;
  if (c == '/')
    return 63;
  if (c == '-')
    return 62;
  if (c == '_')
    return 63;
  return -1;
}

bool i2sInit(int sampleRate)
{
  i2s_config_t cfg = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
      .sample_rate = (uint32_t)sampleRate,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = true,
      .fixed_mclk = 0};
  i2s_pin_config_t pins = {
      .bck_io_num = AUDIO_BCK,
      .ws_io_num = AUDIO_WS,
      .data_out_num = AUDIO_DATA,
      .data_in_num = I2S_PIN_NO_CHANGE};
  esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("i2s_driver_install failed: %d\n", err);
    return false;
  }

  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_pin failed: %d\n", err);
    i2s_driver_uninstall(I2S_PORT);
    return false;
  }

  i2s_zero_dma_buffer(I2S_PORT);
  Serial.println("I2S Ready");
  return true;
}

void i2sDeinit()
{
  i2s_driver_uninstall(I2S_PORT);
}

void i2sWriteSample(int16_t sample)
{
  int16_t buf[2] = {sample, sample};
  size_t written;
  i2s_write(I2S_PORT, buf, sizeof(buf), &written, portMAX_DELAY);
}

static uint16_t readLE16(const uint8_t *p)
{
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t readLE32(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int16_t clampSample(int32_t sample)
{
  if (sample > 32767) return 32767;
  if (sample < -32768) return -32768;
  return (int16_t)sample;
}

// Stream base64 WAV from audio_payload, parse RIFF chunks, tune PCM, then play to I2S.
void streamPlayAudio(WiFiClientSecure &conn, unsigned long deadline)
{
  auto nextCh = [&]() -> int {
    while (millis() < deadline) {
      if (conn.available()) return (uint8_t)conn.read();
      if (!conn.connected()) return -2;
      delay(1);
    }
    return -1;
  };

  auto nextB64Ch = [&]() -> int {
    while (true) {
      int c = nextCh();
      if (c == '\\') {
        int escaped = nextCh();
        if (escaped < 0) return escaped;
        return escaped;
      }
      if (c == '\r' || c == '\n' || c == '\t' || c == ' ')
        continue;
      return c;
    }
  };

  uint8_t decoded[3];
  int decodedLen = 0;
  int decodedIdx = 0;
  bool base64Closed = false;

  auto nextDecodedByte = [&]() -> int {
    if (decodedIdx < decodedLen)
      return decoded[decodedIdx++];

    decodedLen = 0;
    decodedIdx = 0;

    int vals[4] = {0, 0, 0, 0};
    int charsRead = 0;
    int padding = 0;

    while (charsRead < 4) {
      int ch = nextB64Ch();
      if (ch < 0) return ch;
      if (ch == '"') {
        base64Closed = true;
        break;
      }

      if (ch == '=') {
        vals[charsRead++] = 0;
        padding++;
        continue;
      }

      int v = b64val((char)ch);
      if (v < 0)
        continue;

      vals[charsRead++] = v;
    }

    if (charsRead < 2)
      return -3;

    decoded[decodedLen++] = (vals[0] << 2) | (vals[1] >> 4);
    if (charsRead > 2 && padding < 2)
      decoded[decodedLen++] = ((vals[1] & 0x0F) << 4) | (vals[2] >> 2);
    if (charsRead > 3 && padding < 1)
      decoded[decodedLen++] = ((vals[2] & 0x03) << 6) | vals[3];

    return decoded[decodedIdx++];
  };

  auto readBytes = [&](uint8_t *buf, uint32_t len) -> bool {
    for (uint32_t i = 0; i < len; i++) {
      int b = nextDecodedByte();
      if (b < 0) return false;
      buf[i] = (uint8_t)b;
    }
    return true;
  };

  auto skipBytes = [&](uint32_t len) -> bool {
    for (uint32_t i = 0; i < len; i++) {
      if (nextDecodedByte() < 0) return false;
    }
    return true;
  };

  uint8_t riff[12];
  if (!readBytes(riff, sizeof(riff)) || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
    Serial.println("audio_payload is not a RIFF/WAVE file.");
    return;
  }

  uint16_t audioFormat = 0;
  uint16_t channels = 0;
  uint32_t sampleRate = 0;
  uint16_t bitsPerSample = 0;
  uint32_t dataBytes = 0;
  bool sawFmt = false;

  while (!base64Closed) {
    uint8_t chunkHeader[8];
    if (!readBytes(chunkHeader, sizeof(chunkHeader))) {
      Serial.println("Audio payload ended before WAV data chunk.");
      return;
    }

    uint32_t chunkSize = readLE32(chunkHeader + 4);
    bool hasPadByte = (chunkSize & 1) != 0;

    if (memcmp(chunkHeader, "fmt ", 4) == 0) {
      uint8_t fmt[16];
      if (chunkSize < sizeof(fmt) || !readBytes(fmt, sizeof(fmt))) {
        Serial.println("Invalid WAV fmt chunk.");
        return;
      }

      audioFormat = readLE16(fmt);
      channels = readLE16(fmt + 2);
      sampleRate = readLE32(fmt + 4);
      bitsPerSample = readLE16(fmt + 14);

      if (chunkSize > sizeof(fmt) && !skipBytes(chunkSize - sizeof(fmt))) {
        Serial.println("Audio payload ended inside WAV fmt chunk.");
        return;
      }

      sawFmt = true;
    }
    else if (memcmp(chunkHeader, "data", 4) == 0) {
      if (!sawFmt) {
        Serial.println("WAV data chunk arrived before fmt chunk.");
        return;
      }
      dataBytes = chunkSize;
      break;
    }
    else {
      if (!skipBytes(chunkSize)) {
        Serial.println("Audio payload ended inside a WAV metadata chunk.");
        return;
      }
    }

    if (hasPadByte && !skipBytes(1)) {
      Serial.println("Audio payload ended at WAV chunk padding.");
      return;
    }
  }

  if (audioFormat != 1) {
    Serial.printf("Only PCM WAV supported, got format %u.\n", (unsigned)audioFormat);
    return;
  }
  if (bitsPerSample != 16) {
    Serial.printf("Only 16-bit WAV supported, got %u-bit.\n", (unsigned)bitsPerSample);
    return;
  }
  if (channels < 1 || channels > 2) {
    Serial.printf("Only mono/stereo WAV supported, got %u channels.\n", (unsigned)channels);
    return;
  }
  if (sampleRate < 8000 || sampleRate > 48000) {
    Serial.printf("Unsupported WAV sample rate: %lu Hz.\n", (unsigned long)sampleRate);
    return;
  }

  Serial.printf(
      "WAV: %lu Hz, %u ch, %u-bit, fmt %u, data %lu bytes, gain %.2f\n",
      (unsigned long)sampleRate,
      (unsigned)channels,
      (unsigned)bitsPerSample,
      (unsigned)audioFormat,
      (unsigned long)dataBytes,
      AUDIO_GAIN);

  if (!i2sInit((int)sampleRate))
    return;

  uint8_t inFrame[4];
  uint8_t outBuf[1024];
  int outIdx = 0;
  uint32_t dataRead = 0;
  bool endedEarly = false;

  auto flushOut = [&]() {
    if (outIdx > 0) {
      size_t written;
      i2s_write(I2S_PORT, outBuf, outIdx, &written, portMAX_DELAY);
      outIdx = 0;
    }
  };

  while (dataRead + (channels * 2) <= dataBytes) {
    uint32_t frameBytes = channels * 2;
    if (!readBytes(inFrame, frameBytes)) {
      endedEarly = true;
      break;
    }
    dataRead += frameBytes;

    int16_t left = (int16_t)readLE16(inFrame);
    int16_t mixed = left;

    if (channels == 2) {
      int16_t right = (int16_t)readLE16(inFrame + 2);
      mixed = (int16_t)(((int32_t)left + (int32_t)right) / 2);
    }

    int32_t tuned = (int32_t)(mixed * AUDIO_GAIN);
    if (tuned < AUDIO_NOISE_GATE && tuned > -AUDIO_NOISE_GATE)
      tuned = 0;

    int16_t sample = clampSample(tuned);
    outBuf[outIdx++] = sample & 0xFF;
    outBuf[outIdx++] = (sample >> 8) & 0xFF;
    outBuf[outIdx++] = sample & 0xFF;
    outBuf[outIdx++] = (sample >> 8) & 0xFF;

    if (outIdx >= (int)(sizeof(outBuf) - 4))
      flushOut();
  }

  uint32_t leftover = dataBytes - dataRead;
  if (!endedEarly && leftover > 0)
    skipBytes(leftover);

  flushOut();
  i2s_zero_dma_buffer(I2S_PORT);
  i2sDeinit();

  // Keep the HTTPS stream aligned for any JSON fields after audio_payload.
  while (!base64Closed) {
    int ch = nextB64Ch();
    if (ch < 0 || ch == '"') {
      base64Closed = true;
      break;
    }
  }

  if (endedEarly)
    Serial.println("Audio stream ended early.");
  else
    Serial.println("Audio stream finished.");
}

// Stream audio_payload directly from HTTPS connection → I2S, no heap buffer.
// Called immediately after the opening " of the audio_payload value is consumed.
#if 0 // Legacy fixed-header decoder disabled; streamPlayAudio above handles RIFF chunks.
void streamPlayAudioLegacy(WiFiClientSecure &conn, unsigned long deadline)
{
  auto nextCh = [&]() -> int {
    while (millis() < deadline) {
      if (conn.available()) return (uint8_t)conn.read();
      if (!conn.connected()) return -2;
      delay(1);
    }
    return -1;
  };

  auto nextB64Ch = [&]() -> int {
    int c = nextCh();
    if (c == '\\') {
      int escaped = nextCh();
      if (escaped < 0) return escaped;
      return escaped;
    }
    return c;
  };

  // Decode WAV header (44 bytes) from the first 60 b64 chars in the stream
  uint8_t header[WAV_HEADER];
  int hb = 0;
  while (hb < WAV_HEADER)
  {
    int c0 = nextB64Ch(), c1 = nextB64Ch(), c2 = nextB64Ch(), c3 = nextB64Ch();
    if (c0 < 0 || c1 < 0 || c0 == '"' || c1 == '"') {
      Serial.println("Audio payload ended before WAV header was decoded.");
      return;
    }
    int v0 = b64val(c0), v1 = b64val(c1);
    int v2 = (c2 >= 0 && c2 != '"') ? b64val(c2) : -1;
    int v3 = (c3 >= 0 && c3 != '"') ? b64val(c3) : -1;
    if (v0 < 0 || v1 < 0) continue;
    if (hb < WAV_HEADER) header[hb++] = (v0 << 2) | (v1 >> 4);
    if (v2 >= 0 && hb < WAV_HEADER) header[hb++] = ((v1 & 0xF) << 4) | (v2 >> 2);
    if (v3 >= 0 && hb < WAV_HEADER) header[hb++] = ((v2 & 0x3) << 6) | v3;
  }

  int channels      = header[22] | (header[23] << 8);
  int sampleRate    = header[24] | (header[25] << 8) | (header[26] << 16) | (header[27] << 24);
  int bitsPerSample = header[34] | (header[35] << 8);
  if (sampleRate <= 0 || sampleRate > 48000) sampleRate = 22050;
  if (channels < 1 || channels > 2) channels = 1;
  Serial.printf("WAV: %d Hz, %d ch, %d-bit\n", sampleRate, channels, bitsPerSample);
  if (bitsPerSample != 16) { Serial.println("Only 16-bit WAV supported"); return; }

  if (!i2sInit(sampleRate)) return;

  // Decode remaining b64 stream → PCM → I2S in 1 KB batches
  uint8_t outBuf[1024];
  int outIdx = 0;
  uint8_t lo = 0;
  bool hasLo = false, done = false;

  while (!done)
  {
    // Collect 4 b64 chars, stopping at closing "
    int chs[4]; int nc = 0;
    for (int k = 0; k < 4 && !done; k++)
    {
      int ch = nextB64Ch();
      if (ch < 0 || ch == '"') { done = true; break; }
      chs[nc++] = ch;
    }
    if (nc < 2) break;

    int v0 = b64val(chs[0]), v1 = b64val(chs[1]);
    int v2 = nc > 2 ? b64val(chs[2]) : -1;
    int v3 = nc > 3 ? b64val(chs[3]) : -1;
    if (v0 < 0 || v1 < 0) continue;

    uint8_t decoded[3]; int nd = 0;
    decoded[nd++] = (v0 << 2) | (v1 >> 4);
    if (v2 >= 0) decoded[nd++] = ((v1 & 0xF) << 4) | (v2 >> 2);
    if (v3 >= 0) decoded[nd++] = ((v2 & 0x3) << 6) | v3;

    for (int b = 0; b < nd; b++)
    {
      if (!hasLo) { lo = decoded[b]; hasLo = true; continue; }
      uint8_t hi = decoded[b]; hasLo = false;
      outBuf[outIdx++] = lo; outBuf[outIdx++] = hi;
      if (channels == 1) { outBuf[outIdx++] = lo; outBuf[outIdx++] = hi; }
      if (outIdx >= 1020)
      {
        size_t written;
        i2s_write(I2S_PORT, outBuf, outIdx, &written, portMAX_DELAY);
        outIdx = 0;
      }
    }
  }

  if (outIdx > 0) { size_t w; i2s_write(I2S_PORT, outBuf, outIdx, &w, portMAX_DELAY); }
  i2sDeinit();
  Serial.println("Audio stream finished.");
}
#endif

// =======================
// TRANSITION
// =======================
void fastPixelTransition(Emotion from, Emotion to)
{
  for (int i = 0; i <= transitionSteps; i++)
  {
    float progress = i / float(transitionSteps);
    float eased = easeOutCubic(progress);
    int fromMask = (1.0 - eased) * 255;
    int toMask = eased * 255;

    display.clearDisplay();
    drawEmotion(from, t, fromMask, true);
    drawEmotion(to, t + progress, toMask, true);
    display.display();

    t += 0.04;
    delay(transitionDelay);
  }
}

float easeOutCubic(float x)
{
  return 1.0 - pow(1.0 - x, 3.0);
}

// =======================
// DRAWING DISPATCH
// =======================
void drawEmotion(Emotion e, float time, int maskAmount, bool revealMode)
{
  switch (e)
  {
  case SAD:
    drawSadFace(time, maskAmount, revealMode);
    break;
  case HAPPY:
    drawHappyFace(time, maskAmount, revealMode);
    break;
  case STRESSED:
    drawStressedFace(time, maskAmount, revealMode);
    break;
  case SQUINT_HAPPY:
    drawSquintHappyFace(time, maskAmount, revealMode);
    break;
  case SHOCKED:
    drawShockedFace(time, maskAmount, revealMode);
    break;
  }
}

// =======================
// MASKED DRAWING TOOLS
// =======================
uint8_t hashPixel(int x, int y)
{
  uint16_t h = x * 37 + y * 91 + x * y * 13;
  h ^= h >> 8;
  return h & 255;
}

bool allowPixel(int x, int y, int amount, bool revealMode)
{
  if (amount >= 255)
    return true;
  if (amount <= 0)
    return false;
  uint8_t h = hashPixel(x, y);
  return revealMode ? h < amount : h > amount;
}

void mpixel(int x, int y, int amount, bool revealMode)
{
  if (x < 0 || x >= SCREEN_WIDTH || y < 0 || y >= SCREEN_HEIGHT)
    return;
  if (allowPixel(x, y, amount, revealMode))
    display.drawPixel(x, y, SSD1306_WHITE);
}

void mline(int x0, int y0, int x1, int y1, int amount, bool revealMode)
{
  int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  while (true)
  {
    mpixel(x0, y0, amount, revealMode);
    if (x0 == x1 && y0 == y1)
      break;
    int e2 = 2 * err;
    if (e2 >= dy)
    {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx)
    {
      err += dx;
      y0 += sy;
    }
  }
}

void mrect(int x, int y, int w, int h, int amount, bool revealMode)
{
  mline(x, y, x + w, y, amount, revealMode);
  mline(x, y + h, x + w, y + h, amount, revealMode);
  mline(x, y, x, y + h, amount, revealMode);
  mline(x + w, y, x + w, y + h, amount, revealMode);
}

void mfillCircle(int cx, int cy, int r, int amount, bool revealMode)
{
  for (int y = -r; y <= r; y++)
    for (int x = -r; x <= r; x++)
      if (x * x + y * y <= r * r)
        mpixel(cx + x, cy + y, amount, revealMode);
}

void mcircle(int cx, int cy, int r, int amount, bool revealMode)
{
  for (int a = 0; a < 360; a += 4)
  {
    float rad = a * 3.14159 / 180.0;
    mpixel(cx + cos(rad) * r, cy + sin(rad) * r, amount, revealMode);
  }
}

void marcLeft(int cx, int cy, int r, int amount, bool revealMode)
{
  for (int a = 105; a <= 255; a += 3)
  {
    float rad = a * 3.14159 / 180.0;
    mpixel(cx + cos(rad) * r, cy + sin(rad) * r, amount, revealMode);
  }
}

void marcRight(int cx, int cy, int r, int amount, bool revealMode)
{
  for (int a = -75; a <= 75; a += 3)
  {
    float rad = a * 3.14159 / 180.0;
    mpixel(cx + cos(rad) * r, cy + sin(rad) * r, amount, revealMode);
  }
}

void drawParentheses(int ox, int oy, int amount, bool revealMode)
{
  marcLeft(22 + ox, 32 + oy, 23, amount, revealMode);
  marcRight(106 + ox, 32 + oy, 23, amount, revealMode);
}

void drawBlush(int x, int y, int amount, bool revealMode)
{
  mline(x - 5, y + 3, x - 1, y - 3, amount, revealMode);
  mline(x + 1, y + 3, x + 5, y - 3, amount, revealMode);
}

// =======================
// SAD
// =======================
void drawSadFace(float time, int amount, bool revealMode)
{
  int ox = sin(time * 7.0) * 1;
  int oy = sin(time * 2.0) * 1;
  drawSadHands(time, ox, oy, amount, revealMode);
  drawParentheses(ox, oy, amount, revealMode);
  float tear1 = fmod(time * 2.8, 1.0);
  float tear2 = fmod(time * 2.8 + 0.45, 1.0);
  drawCryingEye(39 + ox, 23 + oy, tear1, amount, revealMode);
  drawCryingEye(79 + ox, 23 + oy, tear2, amount, revealMode);
  drawWavySadMouth(64 + ox, 47 + oy, time, amount, revealMode);
}

void drawSadHands(float time, int ox, int oy, int amount, bool revealMode)
{
  int handBounce = sin(time * 5.0) * 2;
  mcircle(8 + ox, 36 + oy + handBounce, 4, amount, revealMode);
  mline(12 + ox, 36 + oy + handBounce, 18 + ox, 34 + oy, amount, revealMode);
  mcircle(120 + ox, 36 + oy - handBounce, 4, amount, revealMode);
  mline(116 + ox, 36 + oy - handBounce, 110 + ox, 34 + oy, amount, revealMode);
}

void drawCryingEye(int x, int y, float tearProgress, int amount, bool revealMode)
{
  mline(x - 9, y - 5, x + 9, y - 5, amount, revealMode);
  mline(x - 7, y - 2, x + 7, y - 2, amount, revealMode);
  mline(x - 4, y - 5, x - 4, y + 8, amount, revealMode);
  mline(x + 4, y - 5, x + 4, y + 8, amount, revealMode);
  int tearLen = 10 + tearProgress * 16;
  mline(x - 4, y + 10, x - 4, y + 10 + tearLen, amount, revealMode);
  mline(x + 4, y + 9, x + 4, y + 9 + tearLen, amount, revealMode);
  int dropY = y + 20 + tearProgress * 28;
  if (dropY < 61)
    mfillCircle(x - 4, dropY, 2, amount, revealMode);
  if (dropY + 6 < 61)
    mfillCircle(x + 4, dropY + 6, 2, amount, revealMode);
}

void drawWavySadMouth(int cx, int cy, float time, int amount, bool revealMode)
{
  int tremble = sin(time * 18.0) * 1;
  mline(cx - 15, cy, cx - 10, cy + 3 + tremble, amount, revealMode);
  mline(cx - 10, cy + 3 + tremble, cx - 5, cy + 1, amount, revealMode);
  mline(cx - 5, cy + 1, cx, cy + 4 - tremble, amount, revealMode);
  mline(cx, cy + 4 - tremble, cx + 5, cy + 1, amount, revealMode);
  mline(cx + 5, cy + 1, cx + 10, cy + 3 + tremble, amount, revealMode);
  mline(cx + 10, cy + 3 + tremble, cx + 15, cy, amount, revealMode);
}

// =======================
// HAPPY
// =======================
void drawHappyFace(float time, int amount, bool revealMode)
{
  int bounce = sin(time * 3.0) * 2;
  int oy = bounce;
  drawParentheses(0, oy, amount, revealMode);
  float blinkPhase = fmod(time * 1.3, 5.0);
  bool blink = blinkPhase > 4.65;
  float winkPhase = fmod(time, 8.0);
  bool wink = winkPhase > 6.4 && winkPhase < 6.75;
  drawHappyEyeSmooth(39, 25 + oy, blink, amount, revealMode);
  drawHappyEyeSmooth(79, 25 + oy, blink || wink, amount, revealMode);
  float mouthOpen = (sin(time * 4.0) + 1.0) * 0.5;
  drawHappyMouthSmooth(64, 43 + oy, mouthOpen, amount, revealMode);
  if ((sin(time * 2.0) + 1.0) * 0.5 > 0.35)
  {
    drawBlush(31, 42 + oy, amount, revealMode);
    drawBlush(91, 42 + oy, amount, revealMode);
  }
}

void drawHappyEyeSmooth(int x, int y, bool closed, int amount, bool revealMode)
{
  if (closed)
    mline(x - 8, y + 4, x + 8, y + 4, amount, revealMode);
  else
  {
    mline(x - 8, y + 8, x, y - 4, amount, revealMode);
    mline(x, y - 4, x + 8, y + 8, amount, revealMode);
  }
}

void drawHappyMouthSmooth(int cx, int cy, float open, int amount, bool revealMode)
{
  int height = 3 + open * 7;
  int width = 8 + open * 3;
  mline(cx - width, cy, cx - 4, cy + height, amount, revealMode);
  mline(cx - 4, cy + height, cx + 4, cy + height, amount, revealMode);
  mline(cx + 4, cy + height, cx + width, cy, amount, revealMode);
}

// =======================
// STRESSED
// =======================
void drawStressedFace(float time, int amount, bool revealMode)
{
  int ox = sin(time * 28.0) * 3;
  int oy = sin(time * 18.0) * 1;
  drawParentheses(ox, oy, amount, revealMode);
  float press = (sin(time * 7.0) + 1.0) * 0.5;
  int pressAmount = press * 5;
  drawGreaterEyeSmooth(39 + ox, 25 + oy, pressAmount, amount, revealMode);
  drawLessEyeSmooth(79 + ox, 25 + oy, pressAmount, amount, revealMode);
  drawStressedMouthSmooth(64 + ox, 47 + oy, time, amount, revealMode);
}

void drawGreaterEyeSmooth(int x, int y, int press, int amount, bool revealMode)
{
  mline(x - 9, y - 8 + press, x + 9, y, amount, revealMode);
  mline(x + 9, y, x - 9, y + 8 - press, amount, revealMode);
}

void drawLessEyeSmooth(int x, int y, int press, int amount, bool revealMode)
{
  mline(x + 9, y - 8 + press, x - 9, y, amount, revealMode);
  mline(x - 9, y, x + 9, y + 8 - press, amount, revealMode);
}

void drawStressedMouthSmooth(int cx, int cy, float time, int amount, bool revealMode)
{
  float tension = (sin(time * 12.0) + 1.0) * 0.5;
  if (tension > 0.45)
  {
    mrect(cx - 13, cy - 4, 26, 8, amount, revealMode);
    for (int i = -10; i <= 10; i += 5)
      mline(cx + i, cy - 4, cx + i + 3, cy + 4, amount, revealMode);
  }
  else
  {
    int shake = sin(time * 25.0) * 1;
    mline(cx - 14, cy + shake, cx + 14, cy - shake, amount, revealMode);
  }
}

// =======================
// SQUINT HAPPY
// =======================
void drawSquintHappyFace(float time, int amount, bool revealMode)
{
  int bounce = sin(time * 2.8) * 2;
  int sway = sin(time * 1.7) * 1;
  int ox = sway, oy = bounce;
  drawParentheses(ox, oy, amount, revealMode);
  float eyeSmile = (sin(time * 3.0) + 1.0) * 0.5;
  int curveLift = eyeSmile * 3;
  drawSquintEye(39 + ox, 26 + oy, curveLift, amount, revealMode);
  drawSquintEye(79 + ox, 26 + oy, curveLift, amount, revealMode);
  float mouthOpen = (sin(time * 3.5) + 1.0) * 0.5;
  drawWideSmile(64 + ox, 43 + oy, mouthOpen, amount, revealMode);
  if (sin(time * 2.5) > -0.2)
  {
    drawBlush(31 + ox, 43 + oy, amount, revealMode);
    drawBlush(91 + ox, 43 + oy, amount, revealMode);
  }
}

void drawSquintEye(int x, int y, int lift, int amount, bool revealMode)
{
  mline(x - 10, y + 2, x - 5, y - 2 - lift, amount, revealMode);
  mline(x - 5, y - 2 - lift, x, y - 3 - lift, amount, revealMode);
  mline(x, y - 3 - lift, x + 5, y - 2 - lift, amount, revealMode);
  mline(x + 5, y - 2 - lift, x + 10, y + 2, amount, revealMode);
}

void drawWideSmile(int cx, int cy, float open, int amount, bool revealMode)
{
  int h = 5 + open * 5;
  int w = 15 + open * 3;
  mline(cx - w, cy, cx - 8, cy + h, amount, revealMode);
  mline(cx - 8, cy + h, cx + 8, cy + h, amount, revealMode);
  mline(cx + 8, cy + h, cx + w, cy, amount, revealMode);
}

// =======================
// SHOCKED
// =======================
void drawShockedFace(float time, int amount, bool revealMode)
{
  int pulse = (sin(time * 6.0) + 1.0) * 2;
  int shake = sin(time * 12.0) * 2;
  int ox = shake, oy = sin(time * 4.0) * 1;
  drawParentheses(ox, oy, amount, revealMode);
  drawSideLookingCircleEye(39 + ox, 26 + oy, 4 + pulse, -2, amount, revealMode);
  drawSideLookingCircleEye(79 + ox, 26 + oy, 4 + pulse, -2, amount, revealMode);
  drawSquareMouth(64 + ox, 47 + oy, 6 + pulse, amount, revealMode);
  drawSoftBlush(31 + ox, 43 + oy, time, amount, revealMode);
  drawSoftBlush(91 + ox, 43 + oy, time, amount, revealMode);
  drawExclamation(108 + ox, 28 + oy, amount, revealMode);
}

void drawSideLookingCircleEye(int x, int y, int r, int lookDir, int amount, bool revealMode)
{
  mcircle(x, y, r, amount, revealMode);
  mfillCircle(x + lookDir, y, 1, amount, revealMode);
  mpixel(x - 1, y - 2, amount, revealMode);
}

void drawSquareMouth(int cx, int cy, int size, int amount, bool revealMode)
{
  mrect(cx - size, cy - size, size * 2, size * 2, amount, revealMode);
}

void drawSoftBlush(int x, int y, float time, int amount, bool revealMode)
{
  float anim = (sin(time * 5.0) + 1.0) * 0.5;
  int offset = anim * 2;
  mline(x - 4, y + offset, x - 1, y - offset, amount, revealMode);
  mline(x + 1, y + offset, x + 4, y - offset, amount, revealMode);
}

void drawExclamation(int x, int y, int amount, bool revealMode)
{
  mline(x, y - 8, x, y + 6, amount, revealMode);
  mline(x + 5, y - 8, x + 5, y + 6, amount, revealMode);
  mfillCircle(x, y + 10, 1, amount, revealMode);
  mfillCircle(x + 5, y + 10, 1, amount, revealMode);
}
