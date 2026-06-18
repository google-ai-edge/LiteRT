// Copyright 2026 Google LLC.
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

#[macro_export]
macro_rules! call_check_status {
    // This matcher captures the function call as an expression (`$call`)
    // and the error cause as a path (`$error_cause`).
    ($call:expr, $error_cause:path) => {
        // The code that will be generated
        let status = $call;
        if status != $crate::bindings::LiteRtStatus_kLiteRtStatusOk {
            return Err($crate::Error::new($error_cause, status));
        }
    };
}
