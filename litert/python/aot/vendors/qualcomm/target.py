"""Compilation target for Qualcomm SOCs."""

import dataclasses
import enum
from typing import Any

from litert.python.aot.core import types

_QUALCOMM_BACKEND_ID = "qualcomm"


# TODO(weiyiw): Generate this from supported_soc.csv.
class SocModel(enum.StrEnum):
  """Qualcomm SOC model."""

  ALL = "ALL"

  SA8255 = "SA8255"
  SA8295 = "SA8295"
  SM8350 = "SM8350"
  SM8450 = "SM8450"
  SM8550 = "SM8550"
  SM8650 = "SM8650"
  SM8750 = "SM8750"


class SocManufacturer(enum.StrEnum):
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
