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

    private val noisePackages = hashSetOf(
        "com.android.systemui",
        "com.samsung.android.honeyboard",
        "android",
        "com.samsung.android.service.aircommand",
        "com.samsung.android.game.gametools",
        "com.google.android.googlequicksearchbox"
    )

    private val launcherPackages = hashSetOf(
        "com.sec.android.app.launcher",
        "com.google.android.apps.nexuslauncher"
    )

    override fun onAccessibilityEvent(event: AccessibilityEvent) {
        if (event.eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) return

        val pkg = event.packageName?.toString() ?: return
        val title = event.className?.toString() ?: ""

        if (noisePackages.contains(pkg)) return

        if (pkg == currentApp) return

        val now = System.currentTimeMillis()

        if (currentApp.isNotEmpty() && !launcherPackages.contains(currentApp)) {
            val duration = (now - currentAppStartTime) / 1000.0

            if (duration > 0.5) {
                savePcStyleLog("app_close", currentApp, currentTitle, duration)
            }
        }

        startRunnable?.let { handler.removeCallbacks(it) }

        currentApp = pkg
        currentTitle = title
        currentAppStartTime = now

        val appToLog = currentApp
        val titleToLog = currentTitle

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
