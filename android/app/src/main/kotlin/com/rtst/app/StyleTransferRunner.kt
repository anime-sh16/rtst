package com.rtst.app

import android.graphics.Bitmap
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class StyleTransferRunner(ptePath: String) {

    private val module: Module = Module.load(ptePath)

    fun stylize(bitmap: Bitmap): Bitmap {
        // Get dimensions
        val width = bitmap.width
        val height = bitmap.height

        // Preprocess — Bitmap to float tensor
        val inputData = preprocessBitmap(bitmap, width, height)

        // Create ExecuTorch tensor with shape [1, 3, height, width]
        val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, height.toLong(), width.toLong()))

        // Run inference
        val outputs = module.forward(EValue.from(inputTensor))

        // Get output float array
        val outputData = outputs[0].toTensor().dataAsFloatArray

        // Postprocess — float tensor back to Bitmap
        return postprocessToBitmap(outputData, width, height)
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
            pixelsFloat[0 * channelSize + i] = r
            pixelsFloat[1 * channelSize + i] = g
            pixelsFloat[2 * channelSize + i] = b
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
