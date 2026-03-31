package com.rtst.app

import org.json.JSONArray
import org.json.JSONObject

/**
 * Holds all benchmark measurements for a single model run.
 */
data class BenchmarkResult(
    // Model info
    val tag: String,
    val backend: String,
    val inputWidth: Int,
    val inputHeight: Int,

    // Benchmark config
    val warmupIters: Int,
    val measureIters: Int,

    // Latency (ms) — one entry per measured iteration
    val latenciesMs: List<Double>,
    val meanLatencyMs: Double,
    val minLatencyMs: Double,
    val maxLatencyMs: Double,
    val p95LatencyMs: Double,

    // Memory (MB)
    val nativeHeapBeforeMb: Double,
    val nativeHeapAfterMb: Double,

    // Device info
    val deviceModel: String,
    val androidVersion: String,
    val timestamp: String
) {
    /**
     * Serialize to JSON matching the format export_pipeline.py expects.
     */
    fun toJson(): String {
        val json = JSONObject()
        json.put("tag", tag)
        json.put("backend", backend)
        json.put("input_width", inputWidth)
        json.put("input_height", inputHeight)
        json.put("warmup_iters", warmupIters)
        json.put("measure_iters", measureIters)
        json.put("latency_ms", JSONArray(latenciesMs))
        json.put("mean_latency_ms", meanLatencyMs)
        json.put("min_latency_ms", minLatencyMs)
        json.put("max_latency_ms", maxLatencyMs)
        json.put("p95_latency_ms", p95LatencyMs)
        json.put("native_heap_before_mb", nativeHeapBeforeMb)
        json.put("native_heap_after_mb", nativeHeapAfterMb)
        json.put("model_memory_delta_mb", nativeHeapAfterMb - nativeHeapBeforeMb)
        json.put("device_model", deviceModel)
        json.put("android_version", androidVersion)
        json.put("timestamp", timestamp)

        return json.toString(2)
    }

    fun toDisplayString(): String = """
        Backend:  $backend
        Model:    $tag
        Input:    ${inputWidth}x${inputHeight}
        Warmup:   $warmupIters  |  Measured: $measureIters

        Mean:     ${"%.1f".format(meanLatencyMs)} ms
        Min:      ${"%.1f".format(minLatencyMs)} ms
        Max:      ${"%.1f".format(maxLatencyMs)} ms
        P95:      ${"%.1f".format(p95LatencyMs)} ms

        Heap before: ${"%.1f".format(nativeHeapBeforeMb)} MB
        Heap after:  ${"%.1f".format(nativeHeapAfterMb)} MB
        Model delta: ${"%.1f".format(nativeHeapAfterMb - nativeHeapBeforeMb
)} MB

        Device:   $deviceModel (Android $androidVersion)
    """.trimIndent()
}
