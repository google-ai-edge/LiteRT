package com.google.ai.edge.litert.benchmark

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.paddingFromBaseline
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.google.ai.edge.litert.CompiledModel
import kotlin.system.measureTimeMillis

class BenchmarkActivity : ComponentActivity() {

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    val modelPath = getStringIntentExtra("model_path")
    val numberOfIterations = getIntIntentExtra("num_iterations")
    val numberOfInferencesPerIteration = getIntIntentExtra("num_inferences_per_iteration")

    @Suppress("MeasureTimeMillis")
    val timeInMillis = measureTimeMillis {
      runBenchmark(modelPath, numberOfIterations, numberOfInferencesPerIteration)
    }

    // TODO(b/402927237): Use the same format as existing tool go/odml-models-guide.
    val result =
      """
      ===== Benchmark Results ====
      Model: ${modelPath}
      Num iterations: ${numberOfIterations}
      Num inferences per iteration: ${numberOfInferencesPerIteration}
      Total Time: ${timeInMillis} millis
      Average Time per iteration: ${timeInMillis / numberOfIterations} millis
      """

    Log.i(TAG, result)
    setContent { Text(result, modifier = Modifier.paddingFromBaseline(top = 100.dp)) }
  }

  fun runBenchmark(
    modelPath: String,
    numberOfIterations: Int,
    numberOfInferencesPerIteration: Int,
  ) {
    Log.i(TAG, "start")

    for (i in 0..numberOfIterations) {
      val compiledModel = CompiledModel.create(modelPath)
      val inputBuffers = compiledModel.createInputBuffers()
      val outputBuffers = compiledModel.createOutputBuffers()

      Log.i(TAG, "iteration ${i}")

      for (j in 0..numberOfInferencesPerIteration) {
        compiledModel.run(inputBuffers, outputBuffers)
      }

      for (tensorBuffer in inputBuffers) {
        tensorBuffer.close()
      }
      for (tensorBuffer in outputBuffers) {
        tensorBuffer.close()
      }
      compiledModel.close()
    }

    Log.i(TAG, "end")
  }

  fun getStringIntentExtra(name: String) = intent.getStringExtra(name)!!

  fun getIntIntentExtra(name: String) = intent.getIntExtra(name, 0)

  companion object {
    const val TAG = "BenchmarkActivity"
  }
}
