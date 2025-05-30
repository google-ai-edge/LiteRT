VERS_1.0 {
  # Export JNI symbols.
  global:
    Java_*;

  # Hide everything else.
  local:
    *;
};
