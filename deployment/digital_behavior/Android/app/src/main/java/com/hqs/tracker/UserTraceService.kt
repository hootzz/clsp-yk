package com.hqs.tracker

import android.accessibilityservice.AccessibilityService
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors

class UserTraceService : AccessibilityService() {

    private val TAG = "HQSTracker"

    private var currentApp = ""
    private var currentTitle = ""
    private var currentAppStartTime = 0L

    private val handler = Handler(Looper.getMainLooper())
    private var startRunnable: Runnable? = null
    private val logExecutor = Executors.newSingleThreadExecutor()

    // 1. 기존 타이머를 방해하지 못하게 투명인간 취급할 노이즈 (구글 검색창 추가됨)
    private val noisePackages = hashSetOf(
        "com.android.systemui",
        "com.samsung.android.honeyboard",
        "android",
        "com.samsung.android.service.aircommand",
        "com.samsung.android.game.gametools",
        "com.google.android.googlequicksearchbox" // <-- [해결] 유튜브 시간 스틸 방지
    )

    // 2. 홈 화면 (세션 마감용)
    private val launcherPackages = hashSetOf(
        "com.sec.android.app.launcher",
        "com.google.android.apps.nexuslauncher"
    )

    override fun onAccessibilityEvent(event: AccessibilityEvent) {
        if (event.eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) return

        val pkg = event.packageName?.toString() ?: return
        val title = event.className?.toString() ?: ""

        // 시스템 오버레이 노이즈는 철저히 무시
        if (noisePackages.contains(pkg)) return

        // 패키지명이 같으면 내부 이동이므로 무시
        if (pkg == currentApp) return

        val now = System.currentTimeMillis()

        // [종료 로직] 이전 앱이 존재했고, 홈 화면이 아니었다면 마감 처리
        if (currentApp.isNotEmpty() && !launcherPackages.contains(currentApp)) {
            val duration = (now - currentAppStartTime) / 1000.0

            // 0.5초 미만으로 머문 앱은 '유령 앱'이므로 아무 기록도 남기지 않음
            if (duration > 0.5) {
                savePcStyleLog("app_close", currentApp, currentTitle, duration)
            }
        }

        // [유령 삼키기 핵심] 이전에 0.5초를 기다리던 예약이 있다면 취소
        startRunnable?.let { handler.removeCallbacks(it) }

        // 새로운 상태로 업데이트
        currentApp = pkg
        currentTitle = title
        currentAppStartTime = now

        // 비동기 실행 시 변수 값이 바뀌는 것을 막기 위한 고정 변수
        val appToLog = currentApp
        val titleToLog = currentTitle

        // [시작 로직] 홈 화면이 아닌 진짜 앱일 경우에만 0.5초 뒤 시작 예약
        if (!launcherPackages.contains(pkg)) {
            startRunnable = Runnable {
                savePcStyleLog("app_start", appToLog, titleToLog, 0.0)
            }
            handler.postDelayed(startRunnable!!, 500)
        }
    }

    private fun savePcStyleLog(type: String, app: String, title: String, dur: Double) {
        val log = JSONObject().apply {
            put("timestamp", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.getDefault()).format(Date()))
            put("device_type", "android")
            put("app", app)
            put("title", title)
            put("duration_seconds", Math.max(0.0, dur))
            put("event_type", type)
            put("url", null)
        }

        logExecutor.execute {
            try {
                val dir = File(getExternalFilesDir(null), "hqs_data")
                if (!dir.exists()) dir.mkdirs()
                val file = File(dir, "events_${SimpleDateFormat("yyyyMMdd").format(Date())}.jsonl")
                file.appendText(log.toString() + "\n")
            } catch (e: Exception) { Log.e(TAG, "Save Error", e) }
        }
    }

    override fun onInterrupt() {}
}