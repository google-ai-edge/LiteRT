"""Compilation target for MediaTek SOCs."""

import dataclasses
import enum
from typing import Any

from litert.python.aot.core import types

_MEDIATEK_BACKEND_ID = "mediatek"


class SocModel(enum.StrEnum):
  """MediaTek SOC model."""

  ALL = "ALL"

  MT6853 = "mt6853"
  MT6877 = "mt6877"
  MT6878 = "mt6878"
  MT6879 = "mt6879"
  MT6886 = "mt6886"
  MT6893 = "mt6893"
  MT6895 = "mt6895"
  MT6897 = "mt6897"
  MT6983 = "mt6983"
  MT6985 = "mt6985"
  MT6989 = "mt6989"
  MT6991 = "mt6991"


class SocManufacturer(enum.StrEnum):
  """MediaTek SOC manufacturer."""

  MEDIATEK = "MediaTek"


class AndroidOsVersion(enum.StrEnum):
  """Android OS version."""

  ALL = "ALL"

  ANDROID_15 = "ANDROID_15"


@dataclasses.dataclass
class Target(types.Target):
  """Compilation target for MediaTek SOCs."""

  soc_model: SocModel
  soc_manufacturer: SocManufacturer = SocManufacturer.MEDIATEK
  android_os_version: AndroidOsVersion = AndroidOsVersion.ANDROID_15

  @classmethod
  def backend_id(cls) -> str:
    return _MEDIATEK_BACKEND_ID

  def __hash__(self) -> int:
    return hash(
        (self.soc_manufacturer, self.soc_model, self.android_os_version)
    )

  def __eq__(self, other: "Target") -> bool:
    return (
        self.soc_manufacturer == other.soc_manufacturer
        and self.soc_model == other.soc_model
        and self.android_os_version == other.android_os_version
    )

  def __repr__(self) -> str:
    return f"{self.soc_manufacturer.value}_{self.soc_model.value}_{self.android_os_version.value}"

  def flatten(self) -> dict[str, Any]:
    flattend_target = super().flatten()
    flattend_target.update({
        "soc_manufacturer": self.soc_manufacturer.value,
        "soc_model": self.soc_model.value,
        "android_os_version": self.android_os_version.value,
    })
    return flattend_target
