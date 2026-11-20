# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pickle file IO utilities.

Minimal addition for the sru-path-aware-rl training pipeline, which saves
environment and agent configurations as ``.pkl`` files next to their YAML
counterparts (see ``isaaclab_nav_task/scripts/train.py``).
"""

import pickle
from pathlib import Path


def dump_pickle(filename: str, data):
    """Serialize an object to a pickle file.

    Args:
        filename: Path of the file to write. Parent directories are created
            if they do not exist.
        data: Object to serialize (must be picklable).
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filename: str):
    """Load an object from a pickle file.

    Args:
        filename: Path of the file to read.

    Returns:
        The deserialized object.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)
