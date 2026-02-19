#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List


@dataclass
class XcodeConfig:
    "Configuration for Xcode in use."
    path: str
    platform: str
    developer_dir: str = ""
    libraries_dir: str = ""
    frameworks_dir: str = ""
    private_frameworks_dir: str = ""
    dylib_dir: str = ""

    def __post_init__(self):
        self.developer_dir = f"{self.path}/Platforms/{self.platform}/Developer"
        self.libraries_dir = f"{self.developer_dir}/Library"
        self.frameworks_dir = f"{self.libraries_dir}/Frameworks"
        self.private_frameworks_dir = f"{self.libraries_dir}/PrivateFrameworks"
        self.dylib_dir = f"{self.developer_dir}/usr/lib"


@dataclass
class XCTRunnerConfig:
    "Configuration for XCTRunner."
    xcode: XcodeConfig
    name: str = "XCTRunner"
    bundle_identifier: str = ""
    info_plist_path: str = ""
    path: str = ""
    template_path: str = ""

    def __post_init__(self):
        self.app = f"{self.name}.app"
        self.template_path = f"{self.xcode.libraries_dir}/Xcode/Agents/XCTRunner.app"
        self.bundle_identifier = f"com.apple.test.{self.name}"
        self.info_plist_path = f"{self.path}/Info.plist"


@dataclass
class Configuration:
    "Configuration for the generator."
    xctrunner_path: str
    platform: str
    name: str
    xctests: List[str]
    xcode_path: str
    log_output: str = "make_xctrunner.log"
    verbose_logging: bool = False
    xcode: XcodeConfig = None
    xctrunner: XCTRunnerConfig = None

    def __post_init__(self):
        self.xcode = XcodeConfig(path=self.xcode_path, platform=self.platform)
        self.xctrunner = XCTRunnerConfig(
            name=self.name, path=self.xctrunner_path, xcode=self.xcode
        )
