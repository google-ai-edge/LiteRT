#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

inline bool LiteRtHasOpenClSupport() {
#if LITERT_HAS_OPENCL_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasOpenGlSupport() {
#if LITERT_HAS_OPENGL_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasAhwbSupport() {
#if LITERT_HAS_AHWB_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasIonSupport() {
#if LITERT_HAS_ION_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasDmaBufSupport() {
#if LITERT_HAS_DMABUF_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasFastRpcSupport() {
#if LITERT_HAS_FASTRPC_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasSyncFenceSupport() {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  return true;
#else
  return false;
#endif
}

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_
