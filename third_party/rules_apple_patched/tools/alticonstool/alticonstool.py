# Copyright 2021 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import plistlib


def plist_load(path):
  with open(path, mode = "rb") as fp:
    if hasattr(plistlib, "load"):
      return plistlib.load(fp)
    else:
      return plistlib.readPlist(fp)


def plist_write(output_path, data):
  with open(output_path, "wb") as fp:
    if hasattr(plistlib, "dump"):
      plistlib.dump(data, fp)
    else:
      plistlib.writePlist(data, fp)


def iconname_from_filename(fname):
  bname, _ = os.path.splitext(os.path.basename(fname))
  return bname.split("@")[0]


def insert_alticons(plist_data, alticons, device_families):
  alticons_data = {}
  for alticon in alticons:
    alticon_id, _ = os.path.splitext(os.path.basename(alticon))
    alticons_data[alticon_id] = {
      "CFBundleIconFiles": sorted(set(map(iconname_from_filename, os.listdir(alticon)))),
    }
  if "iphone" in device_families:
    plist_data["CFBundleIcons"]["CFBundleAlternateIcons"] = alticons_data
  if "ipad" in device_families:
    plist_data["CFBundleIcons~ipad"]["CFBundleAlternateIcons"] = alticons_data


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_plist", required=True)
  parser.add_argument("--output_plist", required=True)
  parser.add_argument("--families", required=True)
  parser.add_argument("--alticon", action="append", required=True)
  args, _ = parser.parse_known_args(argv)

  plist_data = plist_load(args.input_plist)
  insert_alticons(plist_data, args.alticon, args.families)
  plist_write(args.output_plist, plist_data)

  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
