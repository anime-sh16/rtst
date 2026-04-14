package com.rtst.app

import android.graphics.Bitmap
import android.os.Build
import android.os.Debug
import java.time.Instant

/**
 * Wraps StyleTransferRunner to collect benchmark statistics.
 *
 * Usage:
 *   val result = BenchmarkRunner(runner, bitmap, "johnson_in_vulkan_480x640", "vulkan")
 *       .run(warmupIters = 5, measureIters = 20)
 */
class BenchmarkRunner(
    private val runner: StyleTransferRunner,
    private val inputBitmap: Bitmap,
    private val tag: String,
    private val backend: String
) {

    /**
     * Run the full benchmark: warmup + measured iterations.
     * Returns a BenchmarkResult with all stats.
     */
    fun run(warmupIters: Int = 5, measureIters: Int = 20): BenchmarkResult {

        val allocatedHeapBefore = Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)

        for (i in 0 until warmupIters) {
            runner.stylize(inputBitmap)
        }

        val latencies = mutableListOf<Double>()
        for (i in 0 until measureIters) {
            val before = System.nanoTime()
            runner.stylize(inputBitmap)
            val after = System.nanoTime()
            latencies.add((after - before) / 1_000_000.0)
        }

        val allocatedHeapAfter = Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)

        return BenchmarkResult(
            timestamp = Instant.now().toString(),
            tag = tag,
            backend = backend,
            deviceModel = Build.MODEL,
            androidVersion = Build.VERSION.RELEASE,
            inputWidth = inputBitmap.width,
            inputHeight = inputBitmap.height,
            measureIters = measureIters,
            warmupIters = warmupIters,
            latenciesMs = latencies,
            meanLatencyMs = latencies.average(),
            minLatencyMs = latencies.minOrNull() ?: 0.0,
            maxLatencyMs = latencies.maxOrNull() ?: 0.0,
            p95LatencyMs = latencies.sorted()[(latencies.size * 0.95).toInt()],
            nativeHeapBeforeMb = allocatedHeapBefore,
            nativeHeapAfterMb = allocatedHeapAfter,
            // modelMemoryDeltaMb = allocatedHeapAfter - allocatedHeapBefore
        )
    }
}
