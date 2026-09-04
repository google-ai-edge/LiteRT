#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_POWER_TRACE_HOOK_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_POWER_TRACE_HOOK_H_

namespace litert::google_tensor {

struct PowerTraceContext;

PowerTraceContext* CreatePowerTraceContext();

void DestroyPowerTraceContext(PowerTraceContext* context);

// Invoked when kLiteRtHookTypeRuntimeStart is received.
void HandlePowerRuntimeStart(PowerTraceContext* context);

// Invoked when kLiteRtHookTypeRuntimeStop is received.
void HandlePowerRuntimeStop(PowerTraceContext* context);

// Invoked when kLiteRtHookTypeStopAndProcess is received.
void HandlePowerStopAndProcess(PowerTraceContext* context);

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_POWER_TRACE_HOOK_H_
