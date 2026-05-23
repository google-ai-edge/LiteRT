/*
 * Copyright (C) 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NPU_HAL_HOOK_H
#define NPU_HAL_HOOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * NPU HAL Hook Library (libnpu_hal_hook.so)
 * 
* C API that runs inference using a caller-created OpenVINO infer request
 * and notifies the Android NPU HAL Service about work lifecycle phases.
 *
 * The caller is responsible for creating the OpenVINO core, loading the
 * model, compiling it, and creating an infer request. This library takes
 * the infer request, runs async inference, and sends HAL lifecycle
 * notifications.
 *
 * INFERENCE FLOW:
 *   1. Caller creates ov_core, reads model, compiles, creates infer request
 *   2. Caller calls npu_hal_submit_inference_async() or _sync() with infer_request
 *      -> Hook sends work_requested to HAL
 *      -> Hook starts inference (async or sync)
 *      -> Hook sends work_started to HAL
 *      -> For sync: also sends work_ended before returning
 *   3. Caller calls npu_hal_get_output_tensor() to get the result
 *      -> For async: waits for inference to finish, sends work_ended
 *      -> For sync: inference already done, just retrieves output
 *   4. Caller calls npu_hal_release_context() to free the context
 *   5. Caller frees infer_request, compiled_model, core (in that order)
 *
 * AUTO-CONNECT: The library auto-connects on first call.
 */

/**
 * Connect to the NPU HAL service.
 * Called automatically on first use, but can be called explicitly for early init.
 * 
 * @return 0 on success, -1 on failure
 */
int npu_hal_connect(void);

/**
 * Disconnect from the NPU HAL service.
 * Optional - call at application shutdown for clean exit.
 */
void npu_hal_disconnect(void);

/**
 * Check if connected to HAL service.
 * 
 * @return 1 if connected, 0 if not
 */
int npu_hal_is_connected(void);

/**
 * Phase 1: Notify HAL that work has been requested/queued.
 * Call this when OpenVINO receives an inference request.
 * Auto-connects to HAL if not already connected.
 * 
 * @param uid The UID of the calling process (use getuid())
 * @param job_priority Priority of the job (0-1000, 0 is highest priority)
 * @param original_uid Original UID if attributing to another app, -1 otherwise
 * @return workId (>0) on success, -1 on failure
 */
int32_t npu_hal_work_requested(int32_t uid, int32_t job_priority, int32_t original_uid);

/**
 * Phase 2: Notify HAL that NPU has started execution.
 * Call this when the NPU actually begins processing the inference.
 * 
 * @param uid The UID of the calling process
 * @param work_id Token returned by npu_hal_work_requested
 * @param job_priority Priority of the job (0-1000, 0 is highest priority)
 * @return 0 on success, -1 on failure
 */
int npu_hal_work_started(int32_t uid, int32_t work_id, int32_t job_priority);

/**
 * Phase 3: Notify HAL that NPU has finished execution.
 * Call this when the NPU completes the inference.
 * 
 * @param uid The UID of the calling process
 * @param work_id Token returned by npu_hal_work_requested
 * @param job_priority Priority of the job (0-1000, 0 is highest priority)
 * @return 0 on success, -1 on failure
 */
int npu_hal_work_ended(int32_t uid, int32_t work_id, int32_t job_priority);

/**
 * Callback type for priority change notifications.
 * Called when NPU Manager updates an app's scheduling priority.
 *
 * Only invoked for UIDs that have an active inference context (i.e. the
 * hook's uid or original_uid from a currently in-flight submit). Priority
 * changes for UIDs with no active work are silently filtered out.
 * 
 * @param uid The UID whose priority changed
 * @param new_priority The new priority value (0-1000, 0 is highest)
 * @param has_direct_access Whether the app has direct NPU access
 */
typedef void (*npu_hal_priority_callback)(int32_t uid, int32_t new_priority, int has_direct_access);

/**
 * Register a callback for priority change notifications.
 * When NPU Manager updates priorities via updateSchedulingConfigs(),
 * the callback will be invoked so OpenVINO can reorder its queue.
 *
 * The callback is only forwarded for UIDs that currently have active
 * inferences (submitted but not yet released). This prevents stale
 * notifications for apps the caller is no longer serving.
 * 
 * The callback runs on a separate thread - ensure thread safety!
 * 
 * Usage:
 *   void on_priority_change(int32_t uid, int32_t priority, int direct) {
 *       // Reorder pending inferences based on new priority
 *   }
 *   npu_hal_register_priority_callback(on_priority_change);
 * 
 * @param callback Function to call when priorities change
 * @return 0 on success, -1 on failure
 */
int npu_hal_register_priority_callback(npu_hal_priority_callback callback);

/**
 * Unregister the priority change callback.
 * Stops listening for priority updates from the HAL.
 */
void npu_hal_unregister_priority_callback(void);

/**
 * Check if listening for priority updates.
 * 
 * @return 1 if listening, 0 if not
 */
int npu_hal_is_listening_for_priorities(void);

// ============================================================================
// Inference Orchestration API
// ============================================================================

/**
 * Opaque context for an in-flight inference request.
 * Created by npu_hal_submit_inference_async/sync(), freed by
 * npu_hal_release_context().
 */
typedef struct npu_hal_context npu_hal_context_t;

/**
 * Submit an ASYNC inference request using a caller-created infer request.
 * The caller is responsible for ov_core_create / ov_core_read_model /
 * ov_core_compile_model / ov_compiled_model_create_infer_request.
 * This function:
 *   1. Connects to HAL (if needed)
 *   2. Sends work_requested notification to HAL
 *   3. Calls ov_infer_request_start_async() (returns immediately)
 *   4. Sends work_started notification to HAL
 *
 * The caller must then call npu_hal_get_output_tensor() to block until
 * inference completes and retrieve the result. work_ended is sent at
 * that point.
 *
 * OWNERSHIP: The caller RETAINS ownership of the infer_request. It will
 * NOT be freed by npu_hal_release_context(). The caller must keep the
 * infer_request alive until after npu_hal_release_context() is called.
 *
 * @param ctx             Output: opaque context for this inference (caller must release)
 * @param infer_request   An OpenVINO infer request (ov_infer_request_t*, cast to void*)
 * @param job_priority    Priority of the job (0-1000, 0 is highest priority)
 * @param original_uid    Original UID if attributing to another app, -1 otherwise
 * @return 0 on success, -1 on failure
 */
int npu_hal_submit_inference_async(npu_hal_context_t** ctx, void* infer_request,
                                   int32_t job_priority, int32_t original_uid);

/**
 * Submit a SYNC inference request using a caller-created infer request.
 * Same setup as async, but calls ov_infer_request_infer() which blocks
 * until inference completes. All three HAL notifications (work_requested,
 * work_started, work_ended) are sent before this function returns.
 *
 * The caller then calls npu_hal_get_output_tensor() to retrieve the
 * result (no wait needed — inference is already done).
 *
 * OWNERSHIP: The caller RETAINS ownership of the infer_request. It will
 * NOT be freed by npu_hal_release_context().
 *
 * @param ctx             Output: opaque context for this inference (caller must release)
 * @param infer_request   An OpenVINO infer request (ov_infer_request_t*, cast to void*)
 * @param job_priority    Priority of the job (0-1000, 0 is highest priority)
 * @param original_uid    Original UID if attributing to another app, -1 otherwise
 * @return 0 on success, -1 on failure
 */
int npu_hal_submit_inference_sync(npu_hal_context_t** ctx, void* infer_request,
                                  int32_t job_priority, int32_t original_uid);

/**
 * Wait for an async inference to complete.
 * Blocks until the NPU finishes, then sends work_ended notification to HAL.
 * For sync submissions, this is a no-op (inference already done).
 *
 * After this returns successfully, the caller can access the infer_request
 * directly to retrieve outputs in whatever way they need (by index, by name,
 * multiple outputs, etc.) without going through npu_hal_get_output_tensor().
 *
 * @param ctx  Context from npu_hal_submit_inference_async/sync()
 * @return 0 on success, -1 on failure (inference wait failed)
 */
int npu_hal_wait_inference(npu_hal_context_t* ctx);

/**
 * Get the output tensor from a submitted inference.
 * For async submissions: blocks until the NPU finishes, sends work_ended.
 * For sync submissions: inference is already done, just retrieves the tensor.
 *
 * The returned data pointer is valid until npu_hal_release_context() is called.
 *
 * @param ctx         Context from npu_hal_submit_inference_async/sync()
 * @param data        Output: pointer to the output tensor data
 * @param size_bytes  Output: size of the output data in bytes
 * @return 0 on success, -1 on failure
 */
int npu_hal_get_output_tensor(npu_hal_context_t* ctx, void** data, size_t* size_bytes);

/**
 * Release an inference context and free associated resources.
 * Must be called after npu_hal_get_output_tensor() or on error cleanup.
 *
 * @param ctx Context to release (NULL is safe)
 */
void npu_hal_release_context(npu_hal_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif  // NPU_HAL_HOOK_H
