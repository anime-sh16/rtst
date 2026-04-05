package com.rtst.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.content.Intent
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var emptyText: View
    private lateinit var fab: FloatingActionButton
    private lateinit var fabVideo: FloatingActionButton
    private lateinit var spinnerModel: Spinner
    private var runner: StyleTransferRunner? = null

    private val galleryItems = mutableListOf<GalleryItem>()
    private lateinit var adapter: GalleryAdapter

    private var currentPhotoFile: File? = null
    private var currentPhotoUri: Uri? = null

    private val models = ALL_MODELS

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchCameraIntent()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            onPhotoTaken()
        } else {
            Log.e("MainActivity", "Camera capture cancelled or failed")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recyclerView = findViewById(R.id.recyclerGallery)
        emptyText = findViewById(R.id.textEmpty)
        fab = findViewById(R.id.fabCamera)
        spinnerModel = findViewById(R.id.spinnerModel)

        // Model spinner
        val spinnerAdapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            models.map { it.label }
        )
        spinnerModel.adapter = spinnerAdapter
        spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                loadModel(models[position])
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Gallery
        adapter = GalleryAdapter(galleryItems)
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        fabVideo = findViewById(R.id.fabVideo)

        fab.setOnClickListener { launchCamera() }
        fabVideo.setOnClickListener {
            startActivity(Intent(this, CameraActivity::class.java))
        }

        updateEmptyState()
    }

    private fun loadModel(config: ModelConfig) {
        fab.isEnabled = false
        lifecycleScope.launch {
            runner = withContext(Dispatchers.Default) {
                val modelFile = assetToFile(config.assetName)
                StyleTransferRunner(modelFile.absolutePath, config.inputHeight, config.inputWidth)
            }
            fab.isEnabled = true
        }
    }

    private fun launchCamera() {
        if (runner == null) return
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            launchCameraIntent()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchCameraIntent() {
        val photosDir = File(cacheDir, "photos").apply { mkdirs() }
        val photoFile = File.createTempFile("capture_", ".jpg", photosDir)
        currentPhotoFile = photoFile
        currentPhotoUri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", photoFile)
        cameraLauncher.launch(currentPhotoUri!!)
    }

    private fun onPhotoTaken() {
        val photoFile = currentPhotoFile ?: return
        val original = BitmapFactory.decodeFile(photoFile.absolutePath) ?: return
        val currentRunner = runner ?: return

        fab.isEnabled = false

        lifecycleScope.launch {
            val (stylized, elapsedMs) = withContext(Dispatchers.Default) {
                val start = System.nanoTime()
                val result = currentRunner.stylize(original)
                val elapsed = (System.nanoTime() - start) / 1_000_000
                Pair(result, elapsed)
            }

            val modelLabel = models[spinnerModel.selectedItemPosition].label
            galleryItems.add(0, GalleryItem(original, stylized, "$modelLabel — ${elapsedMs} ms"))
            adapter.notifyItemInserted(0)
            recyclerView.scrollToPosition(0)
            updateEmptyState()
            fab.isEnabled = true
        }
    }

    private fun updateEmptyState() {
        emptyText.visibility = if (galleryItems.isEmpty()) View.VISIBLE else View.GONE
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
