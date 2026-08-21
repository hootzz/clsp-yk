# HQS Digital Behavior Collector

A collection of tools for recording app-usage behavior on PC and Android devices.  
Collected data is stored in JSONL format and can later be used for LLM-based summary analysis.

---

## Project structure

```
digital_behavior/
├── pc_collector/
│   └── collector.py        # ActivityWatch polling collector for PC
├── Android/                # Android AccessibilityService collector
│   └── app/src/main/java/com/hqs/tracker/
│       ├── MainActivity.kt
│       └── UserTraceService.kt
├── shared/
│   └── models.py           # Shared data models (DeviceInteractionEvent, Session)
└── README.md
```

---

## Collected data format

PC and Android data are stored in a shared JSONL format.

```json
{
  "timestamp": "2026-04-16T12:11:35+09:00",
  "device_type": "pc",
  "app": "chrome.exe",
  "title": "GitHub - HQS",
  "url": "https://github.com/...",
  "duration_seconds": 47.3,
  "event_type": "app_switch"
}
```

| Field | Description |
|---|---|
| `timestamp` | ISO 8601 timestamp |
| `device_type` | `"pc"` or `"android"` |
| `app` | Process name (PC) or package name (Android) |
| `title` | Window title or activity class name |
| `url` | Browser URL (PC + Chrome/Edge only) |
| `duration_seconds` | Time spent in the app in seconds |
| `event_type` | `"app_switch"` / `"app_start"` / `"app_close"` |

---

## PC setup and usage

### 1. Install ActivityWatch

Download and install the Windows version from the [official ActivityWatch website](https://activitywatch.net/).

After ActivityWatch is launched, its API server runs in the background at `http://localhost:5600`.

### 2. Install Python dependencies

```bash
pip install requests schedule
```

### 3. Run the collector

```bash
cd pc_collector
python collector.py
```

Collected data is saved to `./data/pc/events_YYYYMMDD.jsonl`.

### How it works

- Polls ActivityWatch's `aw-watcher-window` bucket every 5 seconds
- Automatically pauses logging while the user is AFK
- When Chrome, Edge, or Firefox is active, additionally collects the URL and tab title from the `aw-watcher-web` bucket
- On app switches, calculates and stores the duration of the previous app session

---

## Android setup and usage

### 1. Requirements

- Android Studio (latest version recommended)
- Android SDK
- Physical Android device (AccessibilityService functionality is limited on emulators)

### 2. Build and install

```bash
cd android
./gradlew assembleDebug
```

Alternatively, open the project in Android Studio and install it on the device using the `Run` button.

### 3. Permissions

After installation, manually grant the following permissions.

**Enable Accessibility Service**
```
Settings → Accessibility → Installed services → HQS User Trace → On
```

**External storage write permission (Android 10 or earlier)**
```
Settings → Apps → HQS Tracker → Permissions → Storage → Allow
```

When the app is launched for the first time, a screen-capture permission dialog appears. Allow it.

### 4. Data location

Collected data is stored on the device.

```
Android/data/com.hqs.tracker/files/hqs_data/events_YYYYMMDD.jsonl
```

Copy to PC via ADB:
```bash
adb pull /sdcard/Android/data/com.hqs.tracker/files/hqs_data/ ./data/android/
```

### How it works

- Detects app switches using `AccessibilityService`'s `TYPE_WINDOW_STATE_CHANGED` events
- Automatically filters noisy packages such as system UI, keyboards, and the Google search box
- Does not record apps used for less than 0.5 seconds, removing transient apps passed during swipes
- Finalizes the previous app session when the home screen is entered

---

## Notes

- The `data/` folder containing actual collected data is excluded through `.gitignore`. Store research-participant data separately.
- The PC collector requires the ActivityWatch server to be running.
- Because the Android app uses AccessibilityService, excluding it from battery optimization is recommended.
