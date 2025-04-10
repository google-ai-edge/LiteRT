package com.google.ai.edge.litert.sample.dummy;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.Nullable;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.ModelProvider;
import com.google.ai.edge.litert.NpuAcceleratorProvider;
import com.google.ai.edge.litert.TensorBuffer;
import com.google.ai.edge.litert.deployment.AiPackModelProvider;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Arrays;
import java.util.List;

/** Main activity for the test app. */
public class MainActivity extends AppCompatActivity {

  private static final String TAG = "MainActivity";

  private static final List<float[]> testInputTensor =
      Arrays.asList(new float[] {1, 2}, new float[] {10, 20});

  private TextView logView;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    setContentView(R.layout.activity_main);
    logView = findViewById(R.id.log_text);
    logView.setMovementMethod(new ScrollingMovementMethod());

    final Bundle bundle = getIntent().getExtras();
    final boolean useNpuAccelerator =
        bundle != null && bundle.getBoolean("use_npu_accelerator", false);
    final boolean useAiPackModel = bundle != null && bundle.getBoolean("use_ai_pack_model", false);

    if (useAiPackModel) {
      findViewById(R.id.run_scenario_btn).setOnClickListener(v -> runScenarioWithAiPackModel());
      runScenarioWithAiPackModel();
    } else {
      findViewById(R.id.run_scenario_btn)
          .setOnClickListener(v -> runScenarioWithBuitinModel(useNpuAccelerator));
      runScenarioWithBuitinModel(useNpuAccelerator);
    }
  }

  private void runScenarioWithAiPackModel() {
    logEvent("Running inference with LiteRt API and AiPack model");
    ListenableFuture<CompiledModel> compiledModelFuture = simpleAiPackModel();
    Futures.addCallback(
        compiledModelFuture,
        new FutureCallback<CompiledModel>() {
          @Override
          public void onSuccess(CompiledModel compiledModel) {
            runInference(compiledModel);
          }

          @Override
          public void onFailure(Throwable t) {
            logEvent("Failed to get compiled model", t);
          }
        },
        directExecutor());
  }

  /** Runs tests with all available delegates. */
  private void runScenarioWithBuitinModel(boolean useNpuAccelerator) {
    logView.setText("Start scenario\n");

    logEvent("Running inference with LiteRt API");

    CompiledModel compiledModel = useNpuAccelerator ? simpleNpuModel() : simpleCpuGpuModel();
    runInference(compiledModel);
  }

  private void runInference(CompiledModel compiledModel) {
    List<TensorBuffer> inputBuffers = compiledModel.createInputBuffers();
    logEvent("Input buffers size: " + inputBuffers.size());
    for (int i = 0; i < inputBuffers.size(); ++i) {
      inputBuffers.get(i).writeFloat(testInputTensor.get(i));
      logEvent("Input[" + i + "]: " + Arrays.toString(testInputTensor.get(i)));
    }

    List<TensorBuffer> outputBuffers = compiledModel.run(inputBuffers);
    logEvent("Output buffers size: " + outputBuffers.size());
    for (int i = 0; i < outputBuffers.size(); ++i) {
      float[] output = outputBuffers.get(i).readFloat();
      logEvent("Output[" + i + "]: " + Arrays.toString(output));
    }

    for (TensorBuffer buffer : inputBuffers) {
      buffer.close();
    }
    for (TensorBuffer buffer : outputBuffers) {
      buffer.close();
    }
    compiledModel.close();
  }

  private CompiledModel simpleCpuGpuModel() {
    return CompiledModel.create(
        getAssets(),
        "simple_model.tflite",
        new CompiledModel.Options(Accelerator.CPU, Accelerator.GPU));
  }

  private CompiledModel simpleNpuModel() {
    NpuAcceleratorProvider npuAcceleratorProvider = new BuiltinNpuAcceleratorProvider(this);
    Environment env = Environment.create(npuAcceleratorProvider);

    return CompiledModel.create(
        getAssets(), "simple_model_npu.tflite", new CompiledModel.Options(Accelerator.NPU), env);
  }

  @Nullable
  private ListenableFuture<CompiledModel> simpleAiPackModel() {
    NpuAcceleratorProvider npuAcceleratorProvider = new BuiltinNpuAcceleratorProvider(this);
    Environment env = Environment.create(npuAcceleratorProvider);

    AiPackModelProvider aiPackModelProvider =
        new AiPackModelProvider(
            this,
            "simple_model_ai_pack",
            "model/simple_model.tflite",
            Accelerator.CPU,
            Accelerator.GPU,
            Accelerator.NPU);
    ListenableFuture<Void> downloadFuture = aiPackModelProvider.downloadFuture(this);
    return Futures.transform(
        downloadFuture,
        unused -> {
          if (aiPackModelProvider.getType() == ModelProvider.Type.FILE) {
            return CompiledModel.create(
                aiPackModelProvider.getPath(),
                new CompiledModel.Options(
                    aiPackModelProvider.getCompatibleAccelerators().toArray(new Accelerator[0])),
                env);
          } else {
            return CompiledModel.create(
                getAssets(),
                aiPackModelProvider.getPath(),
                new CompiledModel.Options(
                    aiPackModelProvider.getCompatibleAccelerators().toArray(new Accelerator[0])),
                env);
          }
        },
        directExecutor());
  }

  // TODO(niuchl): Use ModelSelector when the accelerator registry could return NPU.
  /*
  private CompiledModel createCompiledModelWithNpu() {
    NpuAcceleratorProvider npuAcceleratorProvider = new BuiltinNpuAcceleratorProvider(this);
    Environment env = Environment.create(npuAcceleratorProvider);

    ModelProvider cpuGpuModelProvider =
        ModelProvider.staticModel(
            ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.CPU, Accelerator.GPU);
    ModelProvider npuModelProvider =
        ModelProvider.staticModel(
            ModelProvider.Type.ASSET, "simple_model_npu.tflite", Accelerator.NPU);
    ModelSelector modelSelector = new ModelSelector(cpuGpuModelProvider, npuModelProvider);
    return CompiledModel.create(modelSelector, env, getAssets());
  }
  */

  private void logEvent(String message) {
    logEvent(message, null);
  }

  private void logEvent(String message, @Nullable Throwable throwable) {
    Log.e(TAG, message, throwable);
    logView.append("• ");
    logView.append(String.valueOf(message)); // valueOf converts null to "null"
    logView.append("\n");
    if (throwable != null) {
      logView.append(throwable.getClass().getCanonicalName() + ": " + throwable.getMessage());
      logView.append("\n");
      logView.append(Arrays.toString(throwable.getStackTrace()));
      logView.append("\n");
    }
  }
}
