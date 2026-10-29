"""Un GTK4 de mentira, solo lo que usa ``ui/gear_scene.py``: widgets con señales, grupos de radio
y ``set_active``/``set_text``/``set_selected`` que emiten como los de verdad (solo si cambia el
valor). Sirve para ejecutar el código de la ventana sin pantalla; no dibuja nada."""

from __future__ import annotations

import re
import sys
import types
from contextlib import contextmanager


class Widget:
    def __init__(self, **kw):
        self._handlers: dict[str, list] = {}
        self._children: list = []
        self._sensitive = True
        self._tooltip = ""
        self._classes: list[str] = []
        self.draws = 0
        self._kw = kw
        self._parent = None

    # señales
    def connect(self, name, cb, *args):
        self._handlers.setdefault(name, []).append((cb, args))
        return len(self._handlers[name])

    def emit(self, name, *extra):
        lead = (None,) if name.startswith("notify::") else ()
        for cb, args in list(self._handlers.get(name, [])):
            cb(self, *lead, *extra, *args)

    def handler_block_by_func(self, _f): ...
    def handler_unblock_by_func(self, _f): ...

    # genéricos
    def set_sensitive(self, v):
        self._sensitive = bool(v)

    def get_sensitive(self):
        return self._sensitive

    def set_tooltip_text(self, t):
        self._tooltip = t

    def add_css_class(self, c):
        self._classes.append(c)

    def queue_draw(self):
        self.draws += 1

    def set_vexpand(self, v): ...
    def set_hexpand(self, v): ...
    def set_margin_start(self, v): ...
    def set_margin_end(self, v): ...
    def set_margin_top(self, v): ...
    def set_margin_bottom(self, v): ...
    def set_child(self, c):
        self._children = [c]

    # contenedores
    def append(self, w):
        w._parent = self
        self._children.append(w)

    def remove(self, w):
        self._children.remove(w)
        w._parent = None

    def get_first_child(self):
        return self._children[0] if self._children else None

    def get_next_sibling(self):
        if self._parent is None:
            return None
        sib = self._parent._children
        i = sib.index(self)
        return sib[i + 1] if i + 1 < len(sib) else None

    def walk(self):
        yield self
        for c in self._children:
            yield from c.walk()


class Label(Widget):
    def __init__(self, label="", **kw):
        super().__init__(**kw)
        self._label = label
        self._text = label

    def set_text(self, t):
        self._text = self._label = t

    def get_text(self):
        return self._text

    def set_markup(self, m):
        self._label = m
        self._text = re.sub(r"<[^>]+>", "", m)

    def get_label(self):
        return self._label

    def set_selectable(self, v): ...


class Button(Widget):
    def __init__(self, label="", **kw):
        super().__init__(**kw)
        self.label = label


class ToggleButton(Button):
    def __init__(self, label="", **kw):
        super().__init__(label, **kw)
        self._active = False

    def get_active(self):
        return self._active

    def set_active(self, v):
        v = bool(v)
        if v != self._active:
            self._active = v
            self.emit("toggled")


class CheckButton(ToggleButton):
    def __init__(self, label="", **kw):
        super().__init__(label, **kw)
        self._group: list[CheckButton] = [self]

    def set_group(self, other):
        other._group.append(self)
        self._group = other._group

    def set_active(self, v):
        v = bool(v)
        if v and len(self._group) > 1:
            for b in self._group:
                if b is not self and b._active:
                    b._active = False
                    b.emit("toggled")
        super().set_active(v)


class Entry(Widget):
    def __init__(self, text="", **kw):
        super().__init__(**kw)
        self._text = text

    def get_text(self):
        return self._text

    def set_text(self, t):
        if t != self._text:
            self._text = t
            self.emit("changed")

    def set_width_chars(self, n): ...


class DropDown(Widget):
    @classmethod
    def new_from_strings(cls, strings):
        d = cls()
        d.strings = list(strings)
        d._sel = 0
        return d

    def get_selected(self):
        return self._sel

    def set_selected(self, i):
        if i != self._sel:
            self._sel = i
            self.emit("notify::selected")


class Range(Widget):
    @classmethod
    def new_with_range(cls, *args):
        r = cls()
        r.lo, r.hi = (args[-3], args[-2]) if len(args) >= 3 else (args[0], args[1])
        r._value = r.lo
        return r

    def get_value(self):
        return self._value

    def set_value(self, v):
        if v != self._value:
            self._value = v
            self.emit("value-changed")


class Scale(Range): ...


class SpinButton(Range): ...


class Box(Widget): ...


class Grid(Widget):
    def attach(self, w, *_a):
        self.append(w)


class ScrolledWindow(Widget):
    def set_min_content_width(self, n): ...
    def set_policy(self, *a): ...


class DrawingArea(Widget):
    def set_content_height(self, n): ...

    def set_draw_func(self, f):
        self.draw_func = f


class ApplicationWindow(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.title = kw.get("title")
        self.visible = False
        self.tick = None

    def add_tick_callback(self, f):
        self.tick = f

    def present(self):
        self.visible = True

    def get_visible(self):
        return self.visible

    def get_application(self):
        return self._kw.get("application")


class Orientation:
    HORIZONTAL = 0
    VERTICAL = 1


class PolicyType:
    NEVER = 0
    AUTOMATIC = 1


class FakeCanvas(DrawingArea):
    def __init__(self):
        super().__init__()
        from mechchar.core.viewport import Viewport

        self.viewport = Viewport(scale=28.0, ox=140.0, oy=420.0)
        self.origin_frac = (0.5, 0.5)
        self.overlay = None
        self.drag_target = None


def _glib():
    g = types.SimpleNamespace()
    g.SOURCE_CONTINUE = True
    g.markup_escape_text = lambda s: (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return g


@contextmanager
def installed():
    """Instala ``gi`` y ``mechchar.ui.canvas`` falsos mientras dura el bloque."""
    gtk = types.SimpleNamespace(
        ApplicationWindow=ApplicationWindow,
        Application=Widget,
        Box=Box,
        Label=Label,
        Button=Button,
        ToggleButton=ToggleButton,
        CheckButton=CheckButton,
        Entry=Entry,
        DropDown=DropDown,
        Scale=Scale,
        SpinButton=SpinButton,
        Grid=Grid,
        ScrolledWindow=ScrolledWindow,
        DrawingArea=DrawingArea,
        Orientation=Orientation,
        PolicyType=PolicyType,
        Widget=Widget,
    )
    repo = types.ModuleType("gi.repository")
    repo.Gtk, repo.GLib = gtk, _glib()
    gi = types.ModuleType("gi")
    gi.require_version = lambda *a, **k: None
    gi.repository = repo
    canvas = types.ModuleType("mechchar.ui.canvas")
    canvas.Canvas = FakeCanvas
    names = ["gi", "gi.repository", "mechchar.ui.canvas", "mechchar.ui.gear_scene"]
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules.update({"gi": gi, "gi.repository": repo, "mechchar.ui.canvas": canvas})
    sys.modules.pop("mechchar.ui.gear_scene", None)
    try:
        yield gtk
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod
