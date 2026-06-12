#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_TPU_TILE_HOOK_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_TPU_TILE_HOOK_H_

#include "litert/vendors/c/litert_dispatch.h"

namespace litert::google_tensor {

struct TpuTileTimeContext;

TpuTileTimeContext* CreateTpuTileTimeContext();

void DestroyTpuTileTimeContext(TpuTileTimeContext* context);

void HandleTpuTileTimeRuntimeStart(TpuTileTimeContext* context,
                                   LiteRtDispatchInvocationContext icontext);

void HandleTpuTileTimeRuntimeStop(TpuTileTimeContext* context,
                                  LiteRtDispatchInvocationContext icontext);

void HandleTpuTileTimeStopAndProcess(TpuTileTimeContext* context);

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_TPU_TILE_HOOK_H_
