package com.rtst.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Real-time camera style transfer using CameraX ImageAnalysis.
 *
 * Flow:
 *   CameraX preview → ImageAnalysis callback → StyleTransferRunner → overlay ImageView
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var stylizedOverlay: ImageView
    private lateinit var spinnerModel: Spinner
    private lateinit var btnToggleView: MaterialButton
    private lateinit var textLatency: TextView
    private lateinit var progressLoading: ProgressBar

    private var runner: StyleTransferRunner? = null
    private var showStylized: Boolean = false
    private var cameraStarted: Boolean = false
    private var loadJob: Job? = null

    // Flag to skip frames while a previous one is still being processed
    @Volatile
    private var isProcessing: Boolean = false

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            init()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        previewView = findViewById(R.id.previewView)
        stylizedOverlay = findViewById(R.id.stylizedOverlay)
        spinnerModel = findViewById(R.id.spinnerModel)
        btnToggleView = findViewById(R.id.btnToggleView)
        textLatency = findViewById(R.id.textLatency)
        progressLoading = findViewById(R.id.progressLoading)

        // Start in raw mode — show camera preview, hide stylized overlay
        showStylized = false
        stylizedOverlay.visibility = View.GONE
        previewView.visibility = View.VISIBLE
        btnToggleView.text = "Show Stylized"
        btnToggleView.isEnabled = false

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            init()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun init() {
        startCamera()
        setupModelSpinner()
        setupToggleButton()
    }

    private fun setupModelSpinner() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            ALL_MODELS.map { it.label }
        )
        spinnerModel.adapter = adapter

        spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                loadModel(ALL_MODELS[position])
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Default to the higher-res Vulkan model for camera mode
        val defaultIndex = ALL_MODELS.indexOfFirst { it.inputHeight == 640 && it.backend == "vulkan" }
            .coerceAtLeast(0)
        spinnerModel.setSelection(defaultIndex)
    }

    private fun setupToggleButton() {
        btnToggleView.setOnClickListener {
            showStylized = !showStylized
            if (showStylized) {
                stylizedOverlay.visibility = View.VISIBLE
                previewView.visibility = View.GONE
                btnToggleView.text = "Show Raw"
            } else {
                stylizedOverlay.visibility = View.GONE
                previewView.visibility = View.VISIBLE
                btnToggleView.text = "Show Stylized"
            }
        }
    }

    private fun loadModel(config: ModelConfig) {
        // Cancel any in-flight load
        loadJob?.cancel()
        runner = null
        isProcessing = false

        // UI → loading state
        setLoadingState(true)

        loadJob = lifecycleScope.launch {
            try {
                val newRunner = withContext(Dispatchers.Default) {
                    val modelFile = assetToFile(config.assetName)
                    StyleTransferRunner(modelFile.absolutePath, config.inputHeight, config.inputWidth)
                }
                runner = newRunner
                textLatency.text = "Ready"
            } catch (e: Exception) {
                Log.e("CameraActivity", "Failed to load model: ${config.assetName}", e)
                textLatency.text = "Load failed"
                Toast.makeText(this@CameraActivity, "Failed to load model", Toast.LENGTH_SHORT).show()
            } finally {
                setLoadingState(false)
            }
        }
    }

    private fun setLoadingState(loading: Boolean) {
        progressLoading.visibility = if (loading) View.VISIBLE else View.GONE
        spinnerModel.isEnabled = !loading
        btnToggleView.isEnabled = !loading
        if (loading) {
            textLatency.text = "Loading model..."
        }
    }

    private fun startCamera() {
        if (cameraStarted) return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(Dispatchers.Default.asExecutor()) { imageProxy ->
                processFrame(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            cameraStarted = true
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        // Capture runner locally to avoid NPE from concurrent model switch
        val currentRunner = runner
        if (!showStylized || isProcessing || currentRunner == null) {
            imageProxy.close()
            return
        }
        isProcessing = true

        try {
            val rawBitmap = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            val bitmap = if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            } else {
                rawBitmap
            }
            val start = System.nanoTime()
            val stylized = currentRunner.stylize(bitmap)
            val elapsed = (System.nanoTime() - start) / 1_000_000
            val fps = if (elapsed > 0) 1000.0 / elapsed else 0.0
            runOnUiThread {
                stylizedOverlay.setImageBitmap(stylized)
                textLatency.text = "${elapsed} ms | ${"%.1f".format(fps)} fps"
                isProcessing = false
            }
        } catch (e: Exception) {
            Log.e("CameraActivity", "Frame processing failed", e)
            isProcessing = false
        } finally {
            imageProxy.close()
        }
    }

    private fun assetToFile(assetName: String): File {
        val outFile = File(filesDir, assetName)
        if (!outFile.exists()) {
            assets.open(assetName).use { input ->
                outFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return outFile
    }
}
