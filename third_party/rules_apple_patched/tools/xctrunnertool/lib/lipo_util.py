#!/usr/binary/env python3

import shutil
import logging
from lib.shell import shell


class LipoUtil:
    "Lipo utility class."

    def __init__(self):
        self.lipo_path = shutil.which("lipo")
        self.log = logging.getLogger(__name__)

    def info(self, binary: str) -> str:
        "Returns the lipo info for the given binary."
        cmd = f"{self.lipo_path} -info {binary}"
        return shell(cmd, check_status=False)

    def current_archs(self, binary: str) -> list:
        "Returns the list of architectures in the given binary."
        archs = self.info(binary)
        try:
            return archs.split("is architecture: ")[1].split()
        except IndexError:
            return archs.split("are: ")[1].split()

    def extract_or_thin(self, binary: str, archs: list[str]):
        "Keeps only the given archs in the given binary."
        cmd = [self.lipo_path, binary]
        if len(archs) == 1:
            cmd.extend(["-thin", archs[0]])
        else:
            for arch in archs:
                cmd.extend(["-extract", arch])
        cmd.extend(["-output", binary])
        shell(" ".join(cmd))
