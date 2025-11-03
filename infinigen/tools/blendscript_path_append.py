# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import os
import sys

pwd = os.getcwd()
if pwd not in sys.path:
    sys.path.append(pwd)

vendor = os.path.join(pwd, ".blender_site")
if os.path.isdir(vendor) and vendor not in sys.path:
    sys.path.append(vendor)
