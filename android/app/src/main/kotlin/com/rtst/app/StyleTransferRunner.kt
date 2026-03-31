package com.rtst.app

import android.graphics.Bitmap
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class StyleTransferRunner(
    ptePath: String,
    private val modelHeight: Int = 640,
    private val modelWidth: Int = 480
) {

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val module: Module = Module.load(ptePath)

    fun stylize(bitmap: Bitmap): Bitmap {
        // Resize to model's expected input size
        val resized = centerCrop(bitmap, modelWidth, modelHeight)

        // Preprocess — Bitmap to float tensor
        val inputData = preprocessBitmap(resized, modelWidth, modelHeight)

        // Create ExecuTorch tensor with shape [1, 3, height, width]
        val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, modelHeight.toLong(), modelWidth.toLong()))

        // Run inference
        val outputs = module.forward(EValue.from(inputTensor))

        // Get output float array
        val outputData = outputs[0].toTensor().dataAsFloatArray

        // Postprocess — float tensor back to Bitmap (at model resolution)
        return postprocessToBitmap(outputData, modelWidth, modelHeight)
    }

    private fun centerCrop(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val sourceWidth = source.width
        val sourceHeight = source.height

        // Calculate the scale required to completely fill the target dimensions
        val scaleFactor = maxOf(
            targetWidth.toFloat() / sourceWidth,
            targetHeight.toFloat() / sourceHeight
        )

        val scaledWidth = (sourceWidth * scaleFactor).toInt()
        val scaledHeight = (sourceHeight * scaleFactor).toInt()

        // 1. Scale the image (maintains aspect ratio, covers the target bounding box)
        val scaledBitmap = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true)

        // 2. Calculate offsets to crop exactly from the center
        val xOffset = maxOf(0, (scaledWidth - targetWidth) / 2)
        val yOffset = maxOf(0, (scaledHeight - targetHeight) / 2)

        // 3. Crop the final target size from the center of the scaled image
        val croppedBitmap = Bitmap.createBitmap(scaledBitmap, xOffset, yOffset, targetWidth, targetHeight)

        // 4. Clean up intermediate bitmap to free memory immediately (crucial for ML pipelines)
        if (scaledBitmap != source && scaledBitmap != croppedBitmap) {
            scaledBitmap.recycle()
        }

        return croppedBitmap
    }

    private fun preprocessBitmap(bitmap: Bitmap, width: Int, height: Int): FloatArray {
        val pixels =IntArray(width * height);
        val pixelsFloat = FloatArray(3 * width * height)

        // Get ARGB pixels
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        val channelSize = width * height

        for (i in pixels.indices) {
            val argb = pixels[i]

            // Convert ARGB to RGB
            val r = (argb shr 16 and 0xFF) / 255.0f
            val g = (argb shr 8 and 0xFF) / 255.0f
            val b = (argb and 0xFF) / 255.0f

            // Store in float array as CHW
            pixelsFloat[0 * channelSize + i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            pixelsFloat[1 * channelSize + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            pixelsFloat[2 * channelSize + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }

        return pixelsFloat
    }

    private fun postprocessToBitmap(data: FloatArray, width: Int, height: Int): Bitmap {
        val channelSize = width * height

        val outputPixels = IntArray(width * height)

        for (i in outputPixels.indices) {
            // multiply with 255, clam and convert to int
            val r = (data[0 * channelSize + i] * 255.0f).coerceIn(0.0f, 255.0f).toInt()
            val g = (data[1 * channelSize + i] * 255.0f).coerceIn(0.0f, 255.0f).toInt()
            val b = (data[2 * channelSize + i] * 255.0f).coerceIn(0.0f, 255.0f).toInt()


            // pack into ARGB pixel
            outputPixels[i] = (0xFF shl 24 or (r shl 16) or (g shl 8) or b)
        }

        return Bitmap.createBitmap(outputPixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
