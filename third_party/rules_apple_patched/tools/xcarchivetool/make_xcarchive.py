import argparse
import datetime
import os
import plistlib
import re
import shutil
import tempfile
import zipfile

from dataclasses import dataclass
from pathlib import Path
from tools.wrapper_common import execute


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--info_plist",
    required=True,
    help="The file path to the Info.plist file for the app.",
  )
  parser.add_argument(
    "--bundle",
    required=True,
    help="The file path to the bundle.",
  )
  parser.add_argument(
    "--output",
    required=True,
    help="The file path to the output .xcarchive.",
  )
  parser.add_argument(
    "--dsym",
    required=False,
    action="append",
    help="The file path to a dSYM file which will be packaged in the .xcarchive.",
  )
  parser.add_argument(
    "--linkmap",
    required=False,
    action="append",
    help="The file path to a linkmap file which will be packaged in the .xcarchive.",
  )
  return parser


@dataclass
class _CodeSignInfo:
  authority: str
  team: str


def _get_codesign_info(app_path: str) -> _CodeSignInfo:
  """Returns the codesign information for the given app bundle using codesign tool."""

  _, _, output = execute.execute_and_filter_output([
    "/usr/bin/codesign",
    "-vv",
    "-d",
    app_path,
  ], raise_on_failure=True)

  authority = re.search(r'^Authority=(.*)$', str(output), re.MULTILINE)
  team = re.search(r'^TeamIdentifier=(.*)$', str(output), re.MULTILINE)
  team = team.group(1) if team and team.group(1) != "not set" else None

  return _CodeSignInfo(
    authority=authority.group(1) if authority else None,
    team=team)


def _extract_app_from_ipa(ipa_path: str, output_path: str) -> str:
  """Extracts the .ipa to a temporary directory and copies the .app to the output directory."""

  with tempfile.TemporaryDirectory(prefix="bazel_temp") as temp_dir:
    # Unzip the .ipa to a temporary directory.
    with open(ipa_path, "rb") as ipa_file:
      ipa_zip = zipfile.ZipFile(ipa_file, "r")
      ipa_zip.extractall(temp_dir)
    # Find the .app.
    app_path = next(Path(temp_dir).glob("**/*.app"))
    if not app_path:
      raise Exception(
        "Could not find .app within the .ipa bundle located at: " + ipa_path)
    # Copy the .app to the output directory.
    shutil.copytree(app_path, output_path, dirs_exist_ok=True)


def _main(
    infoplist_path: Path,
    bundle_path: Path,
    output_path: Path,
    dsyms: list,
    linkmaps: list):
  """Creates a .xcarchive from the given bundle.

  <BundleName>.xcarchive
  Products/
    Applications/
    <BundleName>.app
  Info.plist
  PkgInfo
  dSYMs/
    <BundleName>.app.dSYM
  Linkmaps/
    <BundleName>.txt

  Args:
    infoplist_path: The file path to the Info.plist file for the app.
    bundle_path: The file path to the bundle.
    output_path: The file path to the output .xcarchive.
    dsyms: The file paths to the dSYM files which will be packaged in the .xcarchive.
    linkmaps: The file paths to the linkmap files which will be packaged in the .xcarchive.
  """

  # Collect the required information for the .xcarchive plist
  # from the apps Info.plist.
  with open(infoplist_path, "rb") as f:
    infoplist = plistlib.load(f, fmt=plistlib.FMT_BINARY)
    bundle_name = infoplist["CFBundleName"]
    bundle_id = infoplist["CFBundleIdentifier"]
    bundle_version = infoplist["CFBundleVersion"]
    short_version = infoplist["CFBundleShortVersionString"]

  # Create the .xcarchive directory.
  products_dest_dir = os.path.join(output_path, "Products")
  bundle_dest_dir = os.path.join(products_dest_dir, "Applications")
  bundle_dest_path = os.path.join(bundle_dest_dir, bundle_name + ".app")
  os.makedirs(bundle_dest_dir, exist_ok=True)

  # If is an .ipa, extract and copy .app to destination
  # Else Copy the archive contents to the destination.
  if bundle_path.suffix in [".ipa", ".zip"]:
    _extract_app_from_ipa(bundle_path, bundle_dest_path)
  else:
    shutil.copytree(bundle_path, bundle_dest_path,
            copy_function=shutil.copyfile,
            dirs_exist_ok=True)

  # Copy the dSYM files into the .xcarchive.
  dsyms_path = os.path.join(output_path, "dSYMs")
  for dsym in dsyms or []:
    shutil.copytree(
      dsym,
      os.path.join(dsyms_path, os.path.basename(dsym)),
      dirs_exist_ok=True)

  # Copy the linkmap files into the .xcarchive.
  linkmaps_path = os.path.join(output_path, "Linkmaps")
  if linkmaps:
    os.makedirs(linkmaps_path, exist_ok=True)
  for linkmap in linkmaps or []:
    shutil.copyfile(
      linkmap,
      os.path.join(linkmaps_path, os.path.basename(linkmap)))

  # Create the Info.plist for the .xcarchive.
  creation_date = datetime.datetime.now().isoformat()
  app_relative_path = os.path.relpath(bundle_dest_path, products_dest_dir)
  info_plist = {
    "ApplicationProperties": {
      "ApplicationPath": app_relative_path,
      "ArchiveVersion": 2,
      "CFBundleIdentifier": bundle_id,
      "CFBundleShortVersionString": short_version,
      "CFBundleVersion": bundle_version,
    },
    "CreationDate": creation_date,
    "Name": bundle_name,
    "SchemeName": bundle_name,
  }

  # Add the codesigning information to the .xcarchive Info.plist.
  codesign_info = _get_codesign_info(bundle_dest_path)
  if codesign_info.authority:
    info_plist["ApplicationProperties"]["SigningIdentity"] = codesign_info.authority
  if codesign_info.team:
    info_plist["ApplicationProperties"]["Team"] = codesign_info.team

  # Write the .xcarchive Info.plist.
  info_plist_path = os.path.join(output_path, "Info.plist")
  with open(info_plist_path, "wb") as f:
    plistlib.dump(info_plist, f)


if __name__ == "__main__":
  args = _build_parser().parse_args()

  _main(
    Path(args.info_plist),
    Path(args.bundle),
    Path(args.output),
    args.dsym,
    args.linkmap)

