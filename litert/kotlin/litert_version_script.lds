VERS_1.0 {
  # Export JNI symbols.
  global:
    Java_*;
    JNI_OnLoad;
    JNI_OnUnload;

  # Required for libunwind. This is needed if built and then run internally.
    google_find_phdr;

  # Hide everything else.
  local:
    *;
};
