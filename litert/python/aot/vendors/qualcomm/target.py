"""Compilation target for Qualcomm SOCs."""

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


_QUALCOMM_BACKEND_ID = "qualcomm"


# TODO(weiyiw): Generate this from supported_soc.csv.
class SocModel(StrEnum):
  """Qualcomm SOC model."""

  ALL = "ALL"

  SA8255 = "SA8255"
  SA8295 = "SA8295"
  SM8350 = "SM8350"
  SM8450 = "SM8450"
  SM8550 = "SM8550"
  SM8650 = "SM8650"
  SM8750 = "SM8750"


class SocManufacturer(StrEnum):
  """Qualcomm SOC manufacturer."""

  QUALCOMM = "Qualcomm"


@dataclasses.dataclass
class Target(types.Target):
  """Compilation target for Qualcomm SOCs."""

  soc_model: SocModel
  soc_manufacturer: SocManufacturer = SocManufacturer.QUALCOMM

  @classmethod
  def backend_id(cls) -> str:
    return _QUALCOMM_BACKEND_ID

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
