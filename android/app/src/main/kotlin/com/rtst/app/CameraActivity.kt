package com.rtst.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageView
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

    private var runner: StyleTransferRunner? = null
    private var showStylized: Boolean = true

    // Flag to skip frames while a previous one is still being processed
    private var isProcessing: Boolean = false

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            setupModelSpinner()
            setupToggleButton()
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

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            setupModelSpinner()
            setupToggleButton()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
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
        lifecycleScope.launch {
            runner = withContext(Dispatchers.Default) {
                val modelFile = assetToFile(config.assetName)
                StyleTransferRunner(modelFile.absolutePath, config.inputHeight, config.inputWidth)
            }
            startCamera()
        }
    }

    private fun startCamera() {
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
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (!showStylized || isProcessing || runner == null) {
            imageProxy.close()
            return
        }
        isProcessing = true

        try {
            val bitmap = imageProxy.toBitmap()
            val start = System.nanoTime()
            val stylized = runner!!.stylize(bitmap)
            val elapsed = (System.nanoTime() - start) / 1_000_000
            runOnUiThread {
                stylizedOverlay.setImageBitmap(stylized)
                textLatency.text = "${elapsed} ms"
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
