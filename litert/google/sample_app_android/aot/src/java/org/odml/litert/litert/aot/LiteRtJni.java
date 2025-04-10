package org.odml.litert.litert.aot;

import android.content.res.AssetManager;

/** JNI bridge to forward the calls to the native code, where we can invoke the TFLite C API. */
public class LiteRtJni {

  private final LoggingCallback loggingCallback;

  /**
   * This interface gets called when the JNI wants to print a message (used for debugging purposes).
   */
  public interface LoggingCallback {
    void printLogMessage(String message);
  }

  static {
    System.loadLibrary("litert_native");
  }

  public LiteRtJni(LoggingCallback loggingCallback) {
    this.loggingCallback = loggingCallback;
  }

  void sendLogMessage(String message) {
    if (loggingCallback != null) {
      loggingCallback.printLogMessage(message);
    }
  }

  /** Runs inference with the interpreter. */
  public native void runInference(
      AssetManager assetManager, String tfliteModelName, String npuModelName, String nativeLibDir);
}
