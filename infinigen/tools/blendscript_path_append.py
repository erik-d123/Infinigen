# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import os
import sys

pwd = os.getcwd()
sys.path.append(pwd)

# Also allow a local vendor site-packages folder inside the repo
vendor_dir = os.path.join(pwd, ".blender_site")
if os.path.isdir(vendor_dir):
    sys.path.insert(0, vendor_dir)
