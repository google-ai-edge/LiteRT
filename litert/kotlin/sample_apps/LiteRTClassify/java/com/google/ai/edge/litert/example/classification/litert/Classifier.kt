package com.google.ai.edge.litert.example.classification.litert

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.DequantizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.QuantizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.label.TensorLabel

class Classifier(context: Context, modelPath: String, labelsPath: String) {
  private val labels: List<String>
  private val probabilityPostProcessor: TensorProcessor
  private val imageProcessor: ImageProcessor
  private val model: CompiledModel

  init {
    labels = FileUtil.loadLabels(context, labelsPath)
    imageProcessor =
      ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
        .add(NormalizeOp(0.0f, 255.0f))
        .add(QuantizeOp(0f, 0.0f))
        .add(CastOp(DataType.FLOAT32))
        .build()
    probabilityPostProcessor =
      TensorProcessor.Builder().add(DequantizeOp(0f, 0.0f)).add(NormalizeOp(0.0f, 1.0f)).build()

    model =
      CompiledModel.create(
        context.assets,
        modelPath,
        CompiledModel.Options(Accelerator.CPU, Accelerator.GPU),
      )
  }

  fun classify(image: TensorImage): List<Category> {
    val processedImage = imageProcessor.process(image)

    val inputBuffers = model.createInputBuffers()
    val outputBuffers = model.createOutputBuffers()

    inputBuffers[0].writeFloat(processedImage.tensorBuffer.floatArray)
    model.run(inputBuffers, outputBuffers)
    val results = outputBuffers[0].readFloat()
    val probabilityBuffer =
      org.tensorflow.lite.support.tensorbuffer.TensorBuffer.createFixedSize(
          // TODO(niuchl): derive the shape from the model.
          intArrayOf(1, 5),
          DataType.FLOAT32,
        )
        .apply { loadArray(results) }

    inputBuffers.forEach { it.close() }
    outputBuffers.forEach { it.close() }

    // Map of labels and their corresponding probability
    val labels = TensorLabel(labels, probabilityPostProcessor.process(probabilityBuffer))

    // Create a map to access the result based on label
    return labels.mapWithFloatValue.map { Category(it.key, it.value) }
  }

  fun release() {
    model.close()
  }
}
