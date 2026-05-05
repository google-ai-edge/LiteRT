import argparse
from pathlib import Path
import os

def add_common_arguments(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--model", "-m", type=str, help="path to the tflite model", required=True
  )
  parser.add_argument(
      "--output_dir",
      "-o",
      type=Path,
      help="path to output folder",
      required=True,
  )
  parser.add_argument(
      "--hostname", "-H", type=str, help="hostname for adb", default="localhost"
  )
  parser.add_argument(
      "--serial", "-s", type=str, help="serial for adb", required=True
  )
  parser.add_argument(
      "--soc_model", type=str, help="SoC Model (e.g. SM8650)", required=True
  )
  parser.add_argument(
      "--htp_arch", type=str, help="HTP Arch (e.g. V75)", required=True
  )
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level",
  )
  default_qairt_sdk = (
      Path(__file__).resolve().parents[5] / "third_party" / "qairt" / "latest"
  )
  if "LITERT_QAIRT_SDK" in os.environ:
    default_qairt_sdk = Path(os.environ["LITERT_QAIRT_SDK"]) / "latest"
  parser.add_argument(
      "--qairt_sdk",
      type=Path,
      help="Path to qairt sdk folder",
      required=False,
      default=default_qairt_sdk,
  )
