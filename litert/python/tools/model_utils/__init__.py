"""Re-export ModelUtils from litert in third_party/tensorflow directory.

This file is a placeholder for ModelUtils from litert in third_party/tensorflow
directory. This will be removed/replaced once the ModelUtils and litert is fully
moved to third_party/odml/litert.
"""

import sys
from litert.python.google.tools import model_utils  # pylint: disable=g-direct-tensorflow-import

sys.modules[__name__] = model_utils
