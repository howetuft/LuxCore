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

"""Dear PyGui implementation of the PyLuxCore Tools Menu window."""

import dearpygui.dearpygui as dpg


class MenuWindow:
    """Main menu window for PyLuxCore Tools."""

    def __init__(self):
        """Initialize the menu window."""
        self.selected_tool = None
        self.window_width = 260
        self.window_height = 240

    def setup_ui(self):
        """Set up the user interface."""
        # Create the main window
        with dpg.window(
            tag="Primary Window",
            label="PyLuxCore Tools Menu",
            width=self.window_width,
            height=self.window_height,
            no_collapse=True,
            no_resize=True,
        ):
            # Add menu bar
            with dpg.menu_bar():
                with dpg.menu(label="Tools"):
                    dpg.add_menu_item(
                        label="Quit",
                        callback=self._quit,
                        shortcut="Ctrl+Q"
                    )

            # Add buttons with spacing
            dpg.add_button(
                label="NetNode",
                callback=self._clicked_netnode,
                width=-1,
                height=40
            )
            dpg.add_spacer(height=10)

            dpg.add_button(
                label="NetConsole",
                callback=self._clicked_netconsole,
                width=-1,
                height=40
            )
            dpg.add_spacer(height=20)  # Spacer between buttons and quit

            dpg.add_button(
                label="Quit",
                callback=self._quit,
                width=-1,
                height=40
            )

    def _clicked_netnode(self):
        """Handle NetNode button click."""
        self.selected_tool = "NetNode"
        dpg.stop_dearpygui()

    def _clicked_netconsole(self):
        """Handle NetConsole button click."""
        self.selected_tool = "NetConsole"
        dpg.stop_dearpygui()

    def _quit(self):
        """Handle Quit button click."""
        self.selected_tool = None
        dpg.stop_dearpygui()

    def run(self):
        """Run the menu and return the selected tool."""
        self.setup_ui()
        dpg.create_viewport(
            title="PyLuxCore Tools Menu",
            width=self.window_width,
            height=self.window_height,
            resizable=False
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        return self.selected_tool


def run_menu():
    """Run the menu and return the selected tool."""
    menu = MenuWindow()
    return menu.run()
