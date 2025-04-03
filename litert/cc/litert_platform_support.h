#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PLATFORM_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PLATFORM_SUPPORT_H_

#include "litert/c/litert_platform_support.h"

namespace litert {

// Check whether LiteRT has been compiled with OpenCL support.
inline bool HasOpenClSupport() { return LiteRtHasOpenClSupport(); }

// Check whether LiteRT has been compiled with OpenGl support.
inline bool HasOpenGlSupport() { return LiteRtHasOpenGlSupport(); }

// Check whether LiteRT has been compiled with AHWB support.
inline bool HasAhwbSupport() { return LiteRtHasAhwbSupport(); }

// Check whether LiteRT has been compiled with Ion support.
inline bool HasIonSupport() { return LiteRtHasIonSupport(); }

// Check whether LiteRT has been compiled with DMA-BUF support.
inline bool HasDmaBufSupport() { return LiteRtHasDmaBufSupport(); }

// Check whether LiteRT has been compiled with FastRPC support.
inline bool HasFastRpcSupport() { return LiteRtHasFastRpcSupport(); }

// Check whether LiteRT has been compiled with SyncFence support.
inline bool HasSyncFenceSupport() { return LiteRtHasSyncFenceSupport(); }

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PLATFORM_SUPPORT_H_
