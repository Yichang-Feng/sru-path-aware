"""Pytest bootstrap for the navigation tests.

`isaaclab_nav_task` pulls in `isaaclab.utils`, which imports `pxr` at module level - and `pxr` only
exists once an Isaac Sim app has bootstrapped its libraries. Launching the app here (once, headless)
lets the tests import the real modules instead of stubbing half of Isaac Lab.

Run with:
    cd IsaacLab && ./isaaclab.sh -p -m pytest ../sru-navigation-sim/tests -q
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_parser)
# Parse an empty argv on purpose: pytest's own arguments must not reach the app launcher.
_args_cli, _ = _parser.parse_known_args([])
_args_cli.headless = True
_args_cli.enable_cameras = False

_app_launcher = AppLauncher(_args_cli)
simulation_app = _app_launcher.app
