VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;
    TfLite*;
    LiteRt*;
    kLiteRtRuntimeBuiltin;

  # Hide everything else.
  local:
    *;
};
