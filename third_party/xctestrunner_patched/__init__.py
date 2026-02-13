# This file is just here to work around Bazel auto-generating an empty file here.
# What happens is `import xctestrunner` loads this file instead of the
# subdirectory because the root runfiles directory comes before the subdirectory
# on sys.path
import sys
from . import xctestrunner
sys.modules['xctestrunner'] = xctestrunner
