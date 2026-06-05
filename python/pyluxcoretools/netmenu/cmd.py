###############################################################################
# Copyright 1998-2025 by authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""Command-line entry point for PyLuxCore Tools Menu."""

import logging

from .menuwindow_dp import run_menu
from ..netnode import ui as netnode_ui
from ..netconsole import ui as netconsole_ui
from .._utils import loghandler


logger = logging.getLogger(loghandler.loggerName + ".luxcoremenu")


def main():
    """Main entry point for the PyLuxCore Tools Menu."""
    selected_tool = run_menu()

    if selected_tool == "NetNode":
        netnode_ui()
    elif selected_tool == "NetConsole":
        netconsole_ui()
