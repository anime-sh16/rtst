package com.rtst.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var inputImageView: ImageView
    private lateinit var outputImageView: ImageView
    private lateinit var stylizeButton: Button
    private lateinit var latencyText: TextView

    private lateinit var runner: StyleTransferRunner

    private val MODEL_ASSET_NAME = "johnson_bn_mosaic_xnnpack_fp32_640x480_export_mode.pte"

    private val TEST_IMAGE_ASSET_NAME = "flower.jpg"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageView  = findViewById(R.id.imageViewInput)
        outputImageView = findViewById(R.id.imageViewOutput)
        stylizeButton   = findViewById(R.id.buttonStylize)
        latencyText     = findViewById(R.id.textViewLatency)

        val modelFile = assetToFile(MODEL_ASSET_NAME)
        runner = StyleTransferRunner(modelFile.absolutePath)

        val inputBitmap = loadBitmapFromAssets(TEST_IMAGE_ASSET_NAME)
        inputImageView.setImageBitmap(inputBitmap)

        stylizeButton.setOnClickListener {
            onStylizeClicked(inputBitmap)
        }
    }

    private fun onStylizeClicked(input: Bitmap) {
        lifecycleScope.launch {
            stylizeButton.isEnabled = false
            val startMs = System.currentTimeMillis()
            val output: Bitmap = withContext(Dispatchers.Default) { runner.stylize(input) }
            outputImageView.setImageBitmap(output)
            val elapsedMs = System.currentTimeMillis() - startMs
            latencyText.text = "Latency: ${elapsedMs} ms"
            stylizeButton.isEnabled = true
        }
    }

    /**
     * Copies a file from assets/ into the app's internal filesDir.
     * ExecuTorch Module.load() needs a real filesystem path, not an asset stream.
     */
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


    /**
     * Loads a Bitmap from assets/.
     */
    private fun loadBitmapFromAssets(assetName: String): Bitmap {
        return BitmapFactory.decodeStream(assets.open(assetName))
    }
}
