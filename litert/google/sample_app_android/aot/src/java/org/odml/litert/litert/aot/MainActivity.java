package org.odml.litert.litert.aot;

import android.app.Activity;
import android.os.Build;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.Nullable;
import java.util.Arrays;

/** Main activity for the test app. */
public class MainActivity extends Activity {

  private static final String TAG = "MainActivity";

  private LiteRtJni liteRtJni;
  private TextView logView;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    logView = findViewById(R.id.log_text);
    logView.setMovementMethod(new ScrollingMovementMethod());
    findViewById(R.id.run_scenario_btn).setOnClickListener(v -> runScenario());

    runScenario();
  }

  /** Runs tests with all available delegates. */
  private void runScenario() {
    logView.setText("Start scenario\n");
    liteRtJni = new LiteRtJni(this::logEvent);

    logEvent("Running inference with LiteRt API");
    if (Build.BRAND.equals("samsung")) {
      liteRtJni.runInference(
          getAssets(),
          "simple_model_npu.tflite",
          "simple_model_qualcomm.bin",
          getApplicationInfo().nativeLibraryDir);
    } else if (Build.BRAND.equals("google")) {
      liteRtJni.runInference(
          getAssets(),
          "simple_model_npu.tflite",
          "simple_model_google_tensor.bin",
          getApplicationInfo().nativeLibraryDir);
    } else {
      logEvent("Unsupported vendor: " + Build.BRAND);
    }
  }

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
