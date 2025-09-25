# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compilation target for Google Tensor SOCs."""

import dataclasses
import sys
from typing import Any

from litert.python.aot.core import types

# pylint: disable=g-importing-member
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
if sys.version_info >= (3, 11):
  from enum import StrEnum  # pylint: disable=g-importing-member
else:
  from backports.strenum import StrEnum  # pylint: disable=g-importing-member
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top
# pylint: enable=g-importing-member


_GOOGLE_TENSOR_BACKEND_ID = "GOOGLE"


class SocModel(StrEnum):
  """Google Tensor SOC model."""

  ALL = "ALL"

  TENSOR_G3 = "Tensor_G3"
  TENSOR_G4 = "Tensor_G4"
  TENSOR_G5 = "Tensor_G5"


class SocManufacturer(StrEnum):
  """Google Tensor SOC manufacturer."""

  GOOGLE = "Google"


@dataclasses.dataclass
class Target(types.Target):
  """Compilation target for Google Tensor SOCs."""

  soc_model: SocModel
  soc_manufacturer: SocManufacturer = SocManufacturer.GOOGLE

  @classmethod
  def backend_id(cls) -> str:
    return _GOOGLE_TENSOR_BACKEND_ID

  def __hash__(self) -> int:
    return hash((self.soc_manufacturer, self.soc_model))

  def __eq__(self, other: "Target") -> bool:
    return (
        self.soc_manufacturer == other.soc_manufacturer
        and self.soc_model == other.soc_model
    )

  def __repr__(self) -> str:
    return f"{self.soc_manufacturer.value}_{self.soc_model.value}"

  def flatten(self) -> dict[str, Any]:
    flattend_target = super().flatten()
    flattend_target.update({
        "soc_manufacturer": self.soc_manufacturer.value,
        "soc_model": self.soc_model.value,
    })
    return flattend_target
