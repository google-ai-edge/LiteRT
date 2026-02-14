#!/usr/bin/env python3

import logging
import os
from lib.model import Configuration


class Logger:
    "Logger class."

    def __init__(self, config: Configuration, level: int = logging.INFO):
        if config.verbose_logging:
            level = logging.DEBUG

        logging.basicConfig(
            format="%(asctime)s MakeXCTRunner %(levelname)-8s %(message)s",
            level=level,
            datefmt="%Y-%m-%d %H:%M:%S %z",
            filename=config.log_output,
        )

        if config.verbose_logging:
            # Add console logger in addition to a file logger
            console = logging.StreamHandler()
            console.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s MakeXCTRunner %(levelname)-8s %(message)s"
            )
            console.setFormatter(formatter)
            logging.getLogger("").addHandler(console)

    def get(self, name: str) -> logging.Logger:
        "Returns logger with the given name."
        return logging.getLogger(name)
