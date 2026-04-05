package com.rtst.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Benchmark activity — can be launched standalone or via adb intent.
 *
 * adb command:
 *   adb shell am start -n com.rtst.app/.BenchmarkActivity \
 *       --es model_path "/data/local/tmp/rtst_bench/model.pte" \
 *       --es tag "johnson_in_vulkan" --es backend "vulkan" \
 *       --ei warmup 5 --ei iters 20
 *
 * When launched via intent, writes result.json to filesDir and finishes.
 * When launched manually (button), displays results in the UI.
 */
class BenchmarkActivity : AppCompatActivity() {

    private lateinit var outputImageView: ImageView
    private lateinit var runButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var resultsText: TextView
    private lateinit var spinnerModel: Spinner
    private lateinit var editWarmup: EditText
    private lateinit var editMeasureIters: EditText

    /**
     * Each entry: display label → asset filename → backend tag
     * Add more models here as you export them.
     */
    data class ModelConfig(val label: String, val assetName: String, val backend: String)

    companion object {
        val MODELS = listOf(
            ModelConfig("IN / Vulkan", "johnson_in_mosaic_vulkan_fp32_640x480_export_mode.pte", "vulkan"),
            ModelConfig("IN / XNNPACK", "johnson_in_mosaic_xnnpack_fp32_640x480_export_mode.pte", "xnnpack"),
            ModelConfig("BN / Vulkan", "johnson_bn_mosaic_vulkan_fp32_640x480_export_mode.pte", "vulkan"),
            ModelConfig("BN / XNNPACK", "johnson_bn_mosaic_xnnpack_fp32_640x480_export_mode.pte", "xnnpack"),
        )
        const val TEST_IMAGE_ASSET = "flower.jpg"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_benchmark)

        outputImageView = findViewById(R.id.imageViewBenchOutput)
        runButton = findViewById(R.id.buttonRunBenchmark)
        progressBar = findViewById(R.id.progressBarBench)
        resultsText = findViewById(R.id.textViewResults)
        spinnerModel = findViewById(R.id.spinnerModel)
        editWarmup = findViewById(R.id.editWarmup)
        editMeasureIters = findViewById(R.id.editMeasureIters)

        // Populate model spinner
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            MODELS.map { it.label }
        )
        spinnerModel.adapter = adapter

        // Check if launched via adb intent (has model_path extra)
        val intentModelPath = intent.getStringExtra("model_path")

        if (intentModelPath != null) {
            runAutomatedBenchmark(intentModelPath)
        } else {
            runButton.setOnClickListener { runManualBenchmark() }
        }
    }

    /**
     * Automated mode: called by export_pipeline.py via adb intent.
     */
    private fun runAutomatedBenchmark(modelPath: String) {
        lifecycleScope.launch {
            val resultJson = withContext(Dispatchers.Default) {
                val tag = intent.getStringExtra("tag") ?: "unknown"
                val backend = intent.getStringExtra("backend") ?: "unknown"
                val warmup = intent.getIntExtra("warmup", 5)
                val iters = intent.getIntExtra("iters", 20)

                val inputH = intent.getIntExtra("input_h", 640)
                val inputW = intent.getIntExtra("input_w", 480)
                val runner = StyleTransferRunner(modelPath, inputH, inputW)

                val inputBitmap = BitmapFactory.decodeFile(
                    File(File(modelPath).parent, "input.jpg").absolutePath
                )

                val benchResult = BenchmarkRunner(runner, inputBitmap, tag, backend)
                    .run(warmupIters = warmup, measureIters = iters)

                val outputBitmap = runner.stylize(inputBitmap)

                val json = benchResult.toJson()
                File(filesDir, "bench.json").writeText(json)
                // File(File(modelPath).parent, "result.json").writeText(json)

                val outputFile = File(filesDir, "output.jpg")
                outputFile.outputStream().use { out ->
                    outputBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
                }

                json
            }
            finish()
        }
    }

    /**
     * Manual mode: user taps "Run Benchmark" in the UI.
     */
    private fun runManualBenchmark() {
        lifecycleScope.launch {
            runButton.isEnabled = false
            progressBar.visibility = android.view.View.VISIBLE
            resultsText.text = "Running benchmark..."

            // Read UI config
            val selectedModel = MODELS[spinnerModel.selectedItemPosition]
            val warmup = editWarmup.text.toString().toIntOrNull() ?: 5
            val iters = editMeasureIters.text.toString().toIntOrNull() ?: 20

            val (benchResult, outputBitmap) = withContext(Dispatchers.Default) {
                val modelFile = assetToFile(selectedModel.assetName)
                val runner = StyleTransferRunner(modelFile.absolutePath)
                val inputBitmap = BitmapFactory.decodeStream(assets.open(TEST_IMAGE_ASSET))

                val result = BenchmarkRunner(
                    runner, inputBitmap, selectedModel.assetName, selectedModel.backend
                ).run(warmupIters = warmup, measureIters = iters)

                // One extra inference for the display image
                val output = runner.stylize(inputBitmap)
                Pair(result, output)
            }

            outputImageView.setImageBitmap(outputBitmap)

            // TODO: format stats as a readable string instead of raw JSON
            //   e.g.: "Backend: vulkan\nMean: 103.2 ms\nMin: 95.1 ms\nMax: 118.4 ms\n..."
            resultsText.text = benchResult.toString()

            File(filesDir, "result.json").writeText(benchResult.toJson())

            progressBar.visibility = android.view.View.GONE
            runButton.isEnabled = true
        }
    }

    /** Copy asset to filesDir. */
    private fun assetToFile(assetName: String): File {
        val outFile = File(filesDir, assetName)
        if (!outFile.exists()) {
            assets.open(assetName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile
    }
}
