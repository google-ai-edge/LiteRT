#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

inline bool LiteRtSupportsOpenCl() {
#if LITERT_HAS_OPENCL_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtSupportsOpenGl() {
#if LITERT_HAS_OPENGL_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtSupportsAhwb() {
#if LITERT_HAS_AHWB_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtSupportsIon() {
#if LITERT_HAS_ION_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtSupportsDmaBuf() {
#if LITERT_HAS_DMABUF_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtSupportsFastRpc() {
#if LITERT_HAS_FASTRPC_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtSupportsSyncFence() {
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
