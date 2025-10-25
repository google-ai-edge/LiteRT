import importlib
import sys

OLD_MLIR_SO_MODULE = "ai_edge_litert.mlir._mlir_libs._mlir"
NEW_MLIR_SO_MODULE = "ai_edge_litert._mlir"

print(f"Importing correct modules from '{NEW_MLIR_SO_MODULE}'...")

_mlir = importlib.import_module(NEW_MLIR_SO_MODULE)
sys.modules[OLD_MLIR_SO_MODULE] = _mlir
importlib.import_module("ai_edge_litert.mlir._mlir_libs")
for sub_module in dir(_mlir):
  if sub_module.startswith("_"):
    continue

  # if "ir" not in sub_module:
  #     continue
  try:
    old = f"{OLD_MLIR_SO_MODULE}.{sub_module}"
    new = f"{NEW_MLIR_SO_MODULE}.{sub_module}"
    sys.modules[old] = importlib.import_module(new)
  except ModuleNotFoundError:
    continue
