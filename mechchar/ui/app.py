from __future__ import annotations

import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from mechchar.ui.window import MainWindow  # noqa: E402


class App(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="io.github.mechchar")

    def do_activate(self) -> None:
        win = self.props.active_window or MainWindow(self)
        win.present()


def run() -> int:
    return App().run(sys.argv)
