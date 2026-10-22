from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from mechchar.ui.canvas import Canvas  # noqa: E402


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app, title="Mechanical Characters Designer")
        self.set_default_size(1000, 700)
        self.canvas = Canvas()
        self.set_child(self.canvas)
