/*
 * Copyright 2025 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.litert

/** Exception for various LiteRT API errors. */
class LiteRtException(private val status: Status, message: String) : Exception(message) {
  constructor(code: Int, message: String) : this(Status.fromCode(code), message)
}

/** Status codes for LiteRtException. */
enum class Status(val code: Int) {
  // LINT.IfChange(status_codes)
  Ok(0),

  // Generic errors.
  ErrorInvalidArgument(1),
  ErrorMemoryAllocationFailure(2),
  ErrorRuntimeFailure(3),
  ErrorMissingInputTensor(4),
  ErrorUnsupported(5),
  ErrorNotFound(6),
  ErrorTimeoutExpired(7),
  ErrorWrongVersion(8),
  ErrorUnknown(9),
  ErrorAlreadyExists(10),
  ErrorCancelled(11),

  // File and loading related errors.
  ErrorFileIO(500),
  ErrorInvalidFlatbuffer(501),
  ErrorDynamicLoading(502),
  ErrorSerialization(503),
  ErrorCompilation(504),

  // IR related errors.
  ErrorIndexOOB(1000),
  ErrorInvalidIrType(1001),
  ErrorInvalidGraphInvariant(1002),
  ErrorGraphModification(1003),

  // Tool related errors.
  ErrorInvalidToolConfig(1500),

  // Legalization related errors.
  LegalizeNoMatch(2000),
  ErrorInvalidLegalization(2001),

  // Transformation related errors.
  PatternNoMatch(3000),
  ErrorInvalidTransformation(3001);

  // LINT.ThenChange(../../../../../../../../../c/litert_common.h:status_codes)

  companion object {
    /** Returns the [Status] with the given code. */
    fun fromCode(code: Int): Status =
      requireNotNull(values().firstOrNull { it.code == code }) { "Unknown status code: $code" }
  }
}
