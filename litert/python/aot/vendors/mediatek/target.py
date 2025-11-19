"""Compilation target for MediaTek SOCs."""

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


_MEDIATEK_BACKEND_ID = "mediatek"


class SocModel(StrEnum):
  """MediaTek SOC model."""

  ALL = "ALL"

  MT6853 = "MT6853"
  MT6877 = "MT6877"
  MT6878 = "MT6878"
  MT6879 = "MT6879"
  MT6886 = "MT6886"
  MT6893 = "MT6893"
  MT6895 = "MT6895"
  MT6897 = "MT6897"
  MT6983 = "MT6983"
  MT6985 = "MT6985"
  MT6989 = "MT6989"
  MT6991 = "MT6991"
  MT8171 = "MT8171"
  MT8188 = "MT8188"
  MT8189 = "MT8189"


class SocManufacturer(StrEnum):
  """MediaTek SOC manufacturer."""

  MEDIATEK = "MediaTek"


class AndroidOsVersion(StrEnum):
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
