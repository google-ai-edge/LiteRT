#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PLATFORM_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PLATFORM_SUPPORT_H_

#include "litert/c/litert_platform_support.h"

namespace litert {

inline bool SupportsOpenCl() { return LiteRtSupportsOpenCl(); }

inline bool SupportsOpenGl() { return LiteRtSupportsOpenGl(); }

inline bool SupportsAhwb() { return LiteRtSupportsAhwb(); }

inline bool SupportsIon() { return LiteRtSupportsIon(); }

inline bool SupportsDmaBuf() { return LiteRtSupportsDmaBuf(); }

inline bool SupportsFastRpc() { return LiteRtSupportsFastRpc(); }

inline bool SupportsSyncFence() { return LiteRtSupportsSyncFence(); }

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PLATFORM_SUPPORT_H_
