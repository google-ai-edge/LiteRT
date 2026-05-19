VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;

  # Hide everything else.
  local:
    *;
};
