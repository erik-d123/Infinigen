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

# If provided, also expose a conda site-packages to Blender so it can reuse
# the orchestrator's environment for python modules (e.g., scipy/sklearn/fcl)
conda_site = os.environ.get("BLENDER_CONDA_SITE")
if conda_site and os.path.isdir(conda_site) and conda_site not in sys.path:
    sys.path.insert(0, conda_site)
