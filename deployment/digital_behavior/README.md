# HQS Digital Behavior Collector

PC와 Android 기기에서 앱 사용 행동 데이터를 수집하는 도구 모음입니다.  
수집된 데이터는 JSONL 형식으로 저장되며, 이후 LLM 기반 요약 분석에 활용됩니다.

---

## 프로젝트 구조

```
digital_behavior/
├── pc_collector/
│   └── collector.py        # PC용 ActivityWatch 폴링 수집기
├── Android/                # Android AccessibilityService 수집기
│   └── app/src/main/java/com/hqs/tracker/
│       ├── MainActivity.kt
│       └── UserTraceService.kt
├── shared/
│   └── models.py           # 공통 데이터 모델 (DeviceInteractionEvent, Session)
└── README.md
```

---

## 수집 데이터 포맷

PC / Android 공통 JSONL 형식으로 저장됩니다.

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

| 필드 | 설명 |
|---|---|
| `timestamp` | ISO 8601 형식 타임스탬프 |
| `device_type` | `"pc"` 또는 `"android"` |
| `app` | 프로세스명 (PC) 또는 패키지명 (Android) |
| `title` | 창 제목 또는 액티비티 클래스명 |
| `url` | 브라우저 URL (PC + Chrome/Edge만 해당) |
| `duration_seconds` | 해당 앱에서 머문 시간(초) |
| `event_type` | `"app_switch"` / `"app_start"` / `"app_close"` |

---

## PC 설정 및 사용법

### 1. ActivityWatch 설치

[ActivityWatch 공식 사이트](https://activitywatch.net/)에서 Windows용 설치 파일 다운로드 후 설치합니다.

설치 후 ActivityWatch를 실행하면 백그라운드에서 `http://localhost:5600` 으로 API 서버가 뜹니다.

### 2. Python 의존성 설치

```bash
pip install requests schedule
```

### 3. 수집기 실행

```bash
cd pc_collector
python collector.py
```

수집된 데이터는 `./data/pc/events_YYYYMMDD.jsonl` 에 저장됩니다.

### 동작 방식

- ActivityWatch의 `aw-watcher-window` 버킷을 5초마다 폴링
- AFK(자리 비움) 상태 시 자동으로 기록 중단
- 브라우저(Chrome, Edge, Firefox) 사용 시 `aw-watcher-web` 버킷에서 URL과 탭 제목을 추가로 수집
- 앱 전환 감지 시 이전 앱의 체류 시간을 계산해 저장

---

## Android 설정 및 사용법

### 1. 요구 사항

- Android Studio (최신 버전 권장)
- Android SDK
- 실제 Android 기기 (에뮬레이터는 AccessibilityService 동작 제한)

### 2. 빌드 및 설치

```bash
cd android
./gradlew assembleDebug
```

또는 Android Studio에서 프로젝트를 열고 `Run` 버튼으로 기기에 설치합니다.

### 3. 권한 설정

앱 설치 후 다음 권한을 수동으로 허용해야 합니다.

**접근성 서비스 활성화**
```
설정 → 접근성 → 설치된 서비스 → HQS User Trace → 켜기
```

**외부 저장소 쓰기 권한 (Android 10 이하)**
```
설정 → 앱 → HQS Tracker → 권한 → 저장소 → 허용
```

앱을 처음 실행하면 화면 캡처 권한 요청 팝업이 뜹니다. 허용하면 됩니다.

### 4. 데이터 위치

수집된 데이터는 기기 내부에 저장됩니다.

```
Android/data/com.hqs.tracker/files/hqs_data/events_YYYYMMDD.jsonl
```

ADB로 PC에 복사:
```bash
adb pull /sdcard/Android/data/com.hqs.tracker/files/hqs_data/ ./data/android/
```

### 동작 방식

- `AccessibilityService`의 `TYPE_WINDOW_STATE_CHANGED` 이벤트로 앱 전환 감지
- 시스템 UI, 키보드, Google 검색창 등 노이즈 패키지 자동 필터링
- 0.5초 미만 체류한 앱은 기록하지 않음 (스와이프 중 지나치는 앱 제거)
- 홈 화면 진입 시 이전 앱 세션 마감 처리

---

## 주의 사항

- `data/` 폴더(실제 수집 데이터)는 `.gitignore`로 제외되어 있습니다. 연구 참여자 데이터는 별도 관리하세요.
- PC 수집기는 ActivityWatch 서버가 실행 중이어야 동작합니다.
- Android 앱은 접근성 서비스 특성상 배터리 최적화 예외 설정을 권장합니다.
