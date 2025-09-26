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
"""Utility functions for exporting models to AI pack format."""

import itertools
import os
import pathlib
from typing import cast

from litert.python.aot.core import common
from litert.python.aot.core import types
from litert.python.aot.vendors import fallback_backend
from litert.python.aot.vendors.google_tensor import target as google_tensor_target
from litert.python.aot.vendors.mediatek import mediatek_backend
from litert.python.aot.vendors.mediatek import target as mtk_target
from litert.python.aot.vendors.qualcomm import qualcomm_backend
from litert.python.aot.vendors.qualcomm import target as qnn_target

# TODO: b/407453529 - Add unittests.


_DEVICE_TARGETING_CONFIGURATION = """<config:device-targeting-config
    xmlns:config="http://schemas.android.com/apk/config">
{device_groups}
</config:device-targeting-config>"""

_DEVICE_GROUP_TEMPLATE = """    <config:device-group name="{device_group_name}">
{device_selectors}
    </config:device-group>"""

_DEVICE_SELECTOR_TEMPLATE = """        <config:device-selector>
            <config:system-on-chip manufacturer="{soc_man}" model="{soc_model}"/>
        </config:device-selector>"""


def _is_mobile_device_backend(backend: types.Backend):
  target = backend.target
  if backend.id() == qualcomm_backend.QualcommBackend.id():
    target = cast(qnn_target.Target, target)
    # Non Android QNN targets.
    if target.soc_model in (
        qnn_target.SocModel.SA8255,
        qnn_target.SocModel.SA8295,
    ):
      return False
  return True


def _export_model_files_to_ai_pack(
    compiled_models: types.CompilationResult,
    ai_pack_dir: pathlib.Path,
    ai_pack_name: str,
    litert_model_name: str,
    *,
    separate_mtk_ai_pack: bool = True,
):
  """Exports the model tflite files to the AI pack directory structure.

  Args:
    compiled_models: The compiled models to export.
    ai_pack_dir: The directory to export the AI pack to.
    ai_pack_name: The name of the AI pack.
    litert_model_name: The name of the model in the litert format.
    separate_mtk_ai_pack: Whether to separate the MTK AI pack. If True, the main
      AI pack will use the fallback model for MTK targets. The MTK AI pack will
      contain all MTK models, and empty directories for non-MTK targets.
  """
  fallback_model = None
  for backend, model in compiled_models.models_with_backend:
    if backend.target_id == fallback_backend.FallbackBackend.id():
      fallback_model = model
  assert fallback_model is not None, 'Fallback model is required.'

  model_export_dir = ai_pack_dir / ai_pack_name / 'src/main/assets'
  os.makedirs(model_export_dir, exist_ok=True)
  for backend, model in compiled_models.models_with_backend:
    if not _is_mobile_device_backend(backend):
      continue
    target_id = backend.target_id
    backend_id = backend.id()
    if backend_id == fallback_backend.FallbackBackend.id():
      target_id = 'other'
    elif backend_id == mediatek_backend.MediaTekBackend.id():
      target_id = backend.target_id.replace(
          mtk_target.SocManufacturer.MEDIATEK, 'Mediatek'
      )
    group_name = 'model#group_' + target_id
    export_dir = model_export_dir / group_name
    os.makedirs(export_dir, exist_ok=True)
    model_export_path = export_dir / (litert_model_name + common.DOT_TFLITE)
    if (
        separate_mtk_ai_pack
        and backend_id == mediatek_backend.MediaTekBackend.id()
    ):
      # Use the fallback model for MTK targets in main AI pack.
      model_to_export = fallback_model
    else:
      model_to_export = model
    if not model_to_export.in_memory:
      model_to_export.load()
    model_to_export.save(model_export_path, export_only=True)

  if separate_mtk_ai_pack:
    _export_model_files_to_mtk_ai_pack(
        compiled_models=compiled_models,
        ai_pack_dir=ai_pack_dir,
        ai_pack_name=ai_pack_name + '_mtk',
        litert_model_name=litert_model_name + '_mtk',
    )


def _export_model_files_to_mtk_ai_pack(
    compiled_models: types.CompilationResult,
    ai_pack_dir: pathlib.Path,
    ai_pack_name: str,
    litert_model_name: str,
):
  """Exports the model tflite files to the MTK AI pack directory structure."""
  model_export_dir = ai_pack_dir / ai_pack_name / 'src/main/assets'
  os.makedirs(model_export_dir, exist_ok=True)
  for backend, model in compiled_models.models_with_backend:
    if not _is_mobile_device_backend(backend):
      continue
    backend_id = backend.id()
    target_id = backend.target_id
    if backend_id == fallback_backend.FallbackBackend.id():
      target_id = 'other'
    elif backend_id == mediatek_backend.MediaTekBackend.id():
      target_id = backend.target_id.replace(
          mtk_target.SocManufacturer.MEDIATEK, 'Mediatek'
      )
    group_name = 'model#group_' + target_id
    export_dir = model_export_dir / group_name
    os.makedirs(export_dir, exist_ok=True)
    if backend_id != mediatek_backend.MediaTekBackend.id():
      # Skip non-MTK targets, just create a placeholder file.
      placeholder_file = export_dir / 'placeholder.txt'
      placeholder_file.touch()
      continue
    model_export_path = export_dir / (litert_model_name + common.DOT_TFLITE)
    if not model.in_memory:
      model.load()
    model.save(model_export_path, export_only=True)


def _build_targeting_config(compiled_backends: list[types.Backend]) -> str:
  """Builds device-targeting-config in device_targeting_configuration.xml."""
  device_groups = []
  for backend in compiled_backends:
    if not _is_mobile_device_backend(backend):
      continue
    target = backend.target
    device_group = _target_to_ai_pack_info(target)
    if device_group:
      device_groups.append(device_group)
  device_groups = '\n'.join(device_groups)
  return _DEVICE_TARGETING_CONFIGURATION.format(device_groups=device_groups)


def _target_to_ai_pack_info(target: types.Target) -> str | None:
  """Builds the device group used in device_targeting_configuration.xml."""
  if isinstance(target, qnn_target.Target):
    group_name = str(target)
    selector = _process_qnn_target(target)
    device_selectors = [
        _DEVICE_SELECTOR_TEMPLATE.format(soc_man=man, soc_model=model)
        for man, model in selector
    ]
    device_selectors = '\n'.join(device_selectors)
    device_group = _DEVICE_GROUP_TEMPLATE.format(
        device_group_name=group_name, device_selectors=device_selectors
    )
    return device_group
  elif isinstance(target, mtk_target.Target):
    group_name = str(target).replace(
        mtk_target.SocManufacturer.MEDIATEK, 'Mediatek'
    )
    # TODO: b/407453529 - Support MTK SDK Version / OS version in selector.
    selector = _process_mtk_target(target)
    device_selector = _DEVICE_SELECTOR_TEMPLATE.format(
        soc_man=selector[0], soc_model=selector[1]
    )
    device_group = _DEVICE_GROUP_TEMPLATE.format(
        device_group_name=group_name, device_selectors=device_selector
    )
    return device_group
  elif isinstance(target, google_tensor_target.Target):
    group_name = str(target)
    soc_manufacturer, soc_model = _process_google_tensor_target(target)
    device_selector = _DEVICE_SELECTOR_TEMPLATE.format(
        soc_man=soc_manufacturer, soc_model=soc_model
    )
    device_group = _DEVICE_GROUP_TEMPLATE.format(
        device_group_name=group_name, device_selectors=device_selector
    )
    return device_group
  elif isinstance(target, fallback_backend.FallbackTarget):
    # Don't need to have device selector for fallback target.
    return None
  else:
    print('unsupported target ', target)
    return None


# TODO: b/407453529 - Auto-generate this function from CSVs.
def _process_qnn_target(target: qnn_target.Target) -> list[tuple[str, str]]:
  """Returns the list of (manufacturer, model) for the given QNN target."""
  # Play cannot distinguish between Qualcomm and QTI for now.
  manufacturer = ['Qualcomm', 'QTI']
  models = [str(target.soc_model)]
  return list(itertools.product(manufacturer, models))


# TODO: b/407453529 - Auto-generate this function from CSVs.
def _process_mtk_target(
    target: mtk_target.Target,
) -> tuple[str, str]:
  """Returns tuple of (manufacturer, model) for the given MTK target."""
  # Play cannot distinguish between Qualcomm and QTI for now.
  return str(target.soc_manufacturer).replace(
      mtk_target.SocManufacturer.MEDIATEK, 'Mediatek'
  ), str(target.soc_model)


# TODO: b/407453529 - Auto-generate this function from CSVs.
def _process_google_tensor_target(
    target: google_tensor_target.Target,
) -> tuple[str, str]:
  """Returns tuple of (manufacturer, model) for the given Google Tensor target."""
  return str(target.soc_manufacturer), str(target.soc_model).replace('_', ' ')


def _write_targeting_config(
    compiled_models: types.CompilationResult, ai_pack_dir: pathlib.Path
) -> None:
  """Writes device_targeting_configuration.xml for the given compiled models."""
  compiled_backends = [x for x, _ in compiled_models.models_with_backend]
  targeting_config = _build_targeting_config(
      compiled_backends=compiled_backends
  )

  targeting_config_path = ai_pack_dir / 'device_targeting_configuration.xml'
  targeting_config_path.write_text(targeting_config)


def export(
    compiled_models: types.CompilationResult,
    ai_pack_dir: pathlib.Path | str,
    ai_pack_name: str,
    litert_model_name: str,
) -> None:
  """Exports the compiled models to AI pack format.

  This function will export the compiled models to corresponding directory
  structure:

  {ai_pack_dir}/
    AiPackManifest.xml
    device_targeting_configuration.xml
    {ai_pack_name}/src/main/assets/
      model#group_target_1/
        {litert_model_name}.tflite
      model#group_target_2/
        {litert_model_name}.tflite
      model#group_target_3/
        {litert_model_name}.tflite
      model#group_other/
        {litert_model_name}.tflite

  Args:
    compiled_models: The compiled models to export.
    ai_pack_dir: The directory to export the AI pack to.
    ai_pack_name: The name of the AI pack.
    litert_model_name: The name of the model in the litert format.
  """
  if isinstance(ai_pack_dir, str):
    ai_pack_dir = pathlib.Path(ai_pack_dir)

  ai_pack_dir.mkdir(parents=True, exist_ok=True)

  _export_model_files_to_ai_pack(
      compiled_models=compiled_models,
      ai_pack_dir=ai_pack_dir,
      ai_pack_name=ai_pack_name,
      litert_model_name=litert_model_name,
  )
  _write_targeting_config(
      compiled_models=compiled_models, ai_pack_dir=ai_pack_dir
  )
