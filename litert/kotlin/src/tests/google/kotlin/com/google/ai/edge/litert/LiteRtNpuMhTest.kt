package com.google.ai.edge.litert

import android.content.Context
import androidx.test.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.test.runTest
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class LiteRtNpuMhTest {

  private var context: Context? = null

  @Before
  fun setUp() {
    context = InstrumentationRegistry.getContext()
  }

  @Test
  fun e2eNpuFlow() {
    val npuAcceleratorProvider = BuiltinNpuAcceleratorProvider(context!!)
    val env = Environment.create(npuAcceleratorProvider)

    val npuModel =
      CompiledModel.create(context!!.assets, "simple_model_npu.tflite", optionalEnv = env)

    npuModel.use { compiledModel ->
      val inputBuffers = compiledModel.createInputBuffers()
      assertThat(inputBuffers).hasSize(2)
      for (i in 0 until inputBuffers.size) {
        inputBuffers[i].writeFloat(testInputTensors[i])
      }
      val outputBuffers = compiledModel.createOutputBuffers()
      assertThat(outputBuffers).hasSize(1)
      compiledModel.run(inputBuffers, outputBuffers)
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)
      for (i in 0 until output.size) {
        assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
      }

      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
    }
  }

  @Test
  fun e2eNpuFlow_modelSelector() = runTest {
    val npuAcceleratorProvider = BuiltinNpuAcceleratorProvider(context!!)
    val env = Environment.create(npuAcceleratorProvider)

    val cpuGpuModelProvider =
      ModelProvider.staticModel(
        ModelProvider.Type.ASSET,
        "not_exist.tflite",
        Accelerator.CPU,
        Accelerator.GPU,
      )
    val npuModelProvider =
      ModelProvider.staticModel(
        ModelProvider.Type.ASSET,
        "simple_model_npu.tflite",
        Accelerator.NPU,
      )
    val modelProvider = ModelSelector(cpuGpuModelProvider, npuModelProvider).selectModel(env)

    val npuModel =
      CompiledModel.create(
        context!!.assets,
        modelProvider.getPath(),
        CompiledModel.Options(*modelProvider.getCompatibleAccelerators().toTypedArray()),
        optionalEnv = env,
      )

    npuModel.use { compiledModel ->
      val inputBuffers = compiledModel.createInputBuffers()
      assertThat(inputBuffers).hasSize(2)
      for (i in 0 until inputBuffers.size) {
        inputBuffers[i].writeFloat(testInputTensors[i])
      }
      val outputBuffers = compiledModel.createOutputBuffers()
      assertThat(outputBuffers).hasSize(1)
      compiledModel.run(inputBuffers, outputBuffers)
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)
      for (i in 0 until output.size) {
        assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
      }

      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
    }
  }

  companion object {
    private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
    private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
  }
}
