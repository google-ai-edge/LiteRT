// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_CORE_UTIL_PERFETTO_PROFILING_H_
#define THIRD_PARTY_ODML_LITERT_CORE_UTIL_PERFETTO_PROFILING_H_

#define LITERT_PERFETTO_TRACE_EVENT(event_name)
#define LITERT_PERFETTO_TRACE_EVENT_INSTANT(event_name)
#define LITERT_PERFETTO_TRACE_EVENT_ASYNC_START(event_name, event_id)
#define LITERT_PERFETTO_TRACE_EVENT_ASYNC_END(event_id)

namespace litert::internal {
void InitializePerfetto();
}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_CORE_UTIL_PERFETTO_PROFILING_H_
