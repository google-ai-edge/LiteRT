# LiteRT Build Common

This directory contains common build files and definitions for the LiteRT
project.

## Files

*   **export\_litert\_only\_darwin.lds**: Linker script for Darwin-based systems
(e.g., macOS, iOS) to export only the LiteRT symbols.
*   **export\_litert\_only\_linux.lds**: Linker script for Linux-based systems
to export only the LiteRT symbols.
*   **export\_litert\_runtime\_only\_darwin.lds**: Linker script for
Darwin-based systems to export only the LiteRT runtime symbols.
*   **export\_litert\_runtime\_only\_linux.lds**: Linker script for Linux-based
systems to export only the LiteRT runtime symbols.
*   **litert\_build\_defs.bzl**: This file contains common build definitions and
macros for the LiteRT project. It is used to define the build configurations for
different platforms and to create the final build targets.

*   **special\_rule.bzl**: This file contains special build rules for the LiteRT project.
*   **strict.default.bzl**: This file contains strict build rules for the LiteRT project.
*   **tfl\_model\_gen.bzl**: This file contains rules for generating models for the LiteRT project.
