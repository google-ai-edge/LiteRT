VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;
    TfLite*;
    LiteRt*;

  # Hide everything else.
  local:
    *;
};
