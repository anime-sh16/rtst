package com.rtst.app

/**
 * Central registry of all available style-transfer models.
 *
 * Each entry carries the model's expected input dimensions so that
 * [StyleTransferRunner] can be initialised dynamically
 */
data class ModelConfig(
    val label: String,
    val assetName: String,
    val backend: String,
    val inputHeight: Int,
    val inputWidth: Int,
)

val ALL_MODELS: List<ModelConfig> = listOf(
    //  Height = the first spatial dimension your export used (rows).
    //  Width  = the second spatial dimension (columns).
     ModelConfig(
         label = "BN / Vulkan / fp16 320x240",
         assetName = "johnson_bn_mosaic_vulkan_fp16_320x240_export_mode.pte",
         backend = "vulkan",
         inputHeight = 320,
         inputWidth = 240,
     ),
     ModelConfig(
         label = "BN / Vulkan / fp32 640x480",
         assetName = "johnson_bn_mosaic_vulkan_fp32_640x480_export_mode.pte",
         backend = "vulkan",
         inputHeight = 640,
         inputWidth = 480,
     ),
     ModelConfig(
         label = "BN / XNNPACK / fp32 640x480",
         assetName = "johnson_bn_mosaic_xnnpack_fp32_640x480_export_mode.pte",
         backend = "xnnpack",
         inputHeight = 640,
         inputWidth = 480,
     ),
     ModelConfig(
         label = "BN / XNNPACK / int8 640x480",
         assetName = "johnson_bn_mosaic_xnnpack_int8_640x480_export_mode.pte",
         backend = "xnnpack",
         inputHeight = 640,
         inputWidth = 480,
     ),
     ModelConfig(
         label = "BN / XNNPACK / int8 320x240",
         assetName = "johnson_bn_mosaic_xnnpack_int8_320x240_export_mode.pte",
         backend = "xnnpack",
         inputHeight = 320,
         inputWidth = 240,
     ),
     ModelConfig(
         label = "BN / XNNPACK / int8 320x240 -- distilled",
         assetName = "johnson_bn_mosaic_xnnpack_int8_320x240_distilled.pte",
         backend = "xnnpack",
         inputHeight = 320,
         inputWidth = 240,
     ),
)
