#!/usr/bin/env python3

import logging
import subprocess
import os
import shutil


def shell(command: str, check_status: bool = True) -> str:
    "Runs given shell command and returns stdout output."
    log = logging.getLogger(__name__)
    try:
        log.debug("Running shell command: %s", command)
        output = subprocess.run(
            command, shell=True, check=check_status, capture_output=True
        ).stdout
        return output.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        log.error("Shell command failed: %s", e)
        raise e


def cp_r(src, dst):
    "Copies src recursively to dst and chmod with full access."
    os.makedirs(dst, exist_ok=True)  # create dst if it doesn't exist
    shutil.copytree(src, dst, dirs_exist_ok=True)
