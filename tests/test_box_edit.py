"""Mover y redimensionar la caja de engranajes: geometría (GearBox) y editor (BoxEditor)."""

import pytest

from mechchar.core.viewport import Viewport
from mechchar.optim.gearbox import GearBox
from mechchar.ui.box_editor import HANDLE_PX, MIN_SIZE_PX, BoxEditor

BOX = GearBox(0.0, 0.0, 4.0, 2.0)


# --- geometría ---------------------------------------------------------------------------------


def test_translated_keeps_the_size() -> None:
    moved = BOX.translated(1.5, -0.5)
    assert moved == GearBox(1.5, -0.5, 5.5, 1.5)
    assert (moved.width, moved.height) == (BOX.width, BOX.height)


@pytest.mark.parametrize(
    ("handle", "expected"),
    [
        ("e", GearBox(0.0, 0.0, 4.5, 2.0)),
        ("w", GearBox(0.5, 0.0, 4.0, 2.0)),
        ("n", GearBox(0.0, 0.0, 4.0, 2.25)),
        ("s", GearBox(0.0, 0.25, 4.0, 2.0)),
        ("ne", GearBox(0.0, 0.0, 4.5, 2.25)),
        ("sw", GearBox(0.5, 0.25, 4.0, 2.0)),
        ("se", GearBox(0.0, 0.25, 4.5, 2.0)),
        ("nw", GearBox(0.5, 0.0, 4.0, 2.25)),
    ],
)
def test_resized_moves_only_the_grabbed_edges(handle: str, expected: GearBox) -> None:
    assert BOX.resized(handle, 0.5, 0.25, 0.1) == expected


def test_resized_stops_against_the_opposite_edge_instead_of_crossing_it() -> None:
    assert BOX.resized("e", -10.0, 0.0, 0.3) == GearBox(0.0, 0.0, 0.3, 2.0)
    assert BOX.resized("w", 10.0, 0.0, 0.3) == GearBox(3.7, 0.0, 4.0, 2.0)
    assert BOX.resized("n", 0.0, -10.0, 0.3) == GearBox(0.0, 0.0, 4.0, 0.3)
    assert BOX.resized("s", 0.0, 10.0, 0.3) == GearBox(0.0, 1.7, 4.0, 2.0)
    shrunk = BOX.resized("se", -10.0, 10.0, 0.3)
    assert shrunk.width == pytest.approx(0.3) and shrunk.height == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((2.0, 1.0), "move"),
        ((0.05, 1.0), "w"),
        ((3.95, 1.0), "e"),
        ((2.0, 1.95), "n"),
        ((2.0, 0.05), "s"),
        ((0.05, 1.95), "nw"),
        ((3.95, 1.95), "ne"),
        ((0.05, 0.05), "sw"),
        ((3.95, 0.05), "se"),
        ((-0.05, 1.0), "w"),  # un poco afuera todavía agarra el borde
        ((2.0, 2.4), None),
        ((-1.0, -1.0), None),
    ],
)
def test_hit_tells_what_is_grabbed(point: tuple[float, float], expected: str | None) -> None:
    assert BOX.hit(point, 0.1) == expected


def test_hit_on_a_tiny_box_keeps_a_center_to_move_it() -> None:
    tiny = GearBox(0.0, 0.0, 0.3, 0.3)  # la tolerancia (0,2) es mayor que un tercio del lado
    assert tiny.hit((0.15, 0.15), 0.2) == "move"
    assert tiny.hit((0.02, 0.15), 0.2) == "w"
    assert tiny.hit((0.28, 0.15), 0.2) == "e"
    assert tiny.hit((0.02, 0.02), 0.2) == "sw"


# --- editor ------------------------------------------------------------------------------------


def _editor(box: GearBox | None = BOX) -> tuple[BoxEditor, Viewport, list[str]]:
    vp = Viewport()
    vp.scale, vp.ox, vp.oy = 100.0, 50.0, 400.0  # mundo (x, y) -> pantalla (50 + 100x, 400 - 100y)
    ed = BoxEditor(vp)
    ed.box = box
    events: list[str] = []
    ed.on_change = lambda: events.append("change")
    return ed, vp, events


Point = tuple[float, float]


def _drag(ed: BoxEditor, vp: Viewport, start: Point, end: Point) -> None:
    a, b = vp.world_to_screen(*start), vp.world_to_screen(*end)
    assert ed.drag_begin(*a)
    ed.drag_update((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ed.drag_update(*b)
    ed.drag_end()


def test_dragging_outside_draws_a_new_box_as_before() -> None:
    ed, vp, events = _editor()
    _drag(ed, vp, (5.0, 3.0), (7.0, 4.0))
    assert ed.box == GearBox(5.0, 3.0, 7.0, 4.0)
    assert events == ["change"]


def test_dragging_from_inside_moves_the_box() -> None:
    ed, vp, events = _editor()
    _drag(ed, vp, (2.0, 1.0), (3.0, 0.5))
    assert ed.box == pytest.approx(BOX.translated(1.0, -0.5))
    assert events == ["change"]


def test_dragging_an_edge_or_corner_resizes_the_box() -> None:
    ed, vp, _ = _editor()
    _drag(ed, vp, (4.0, 1.0), (5.0, 1.0))  # borde derecho
    assert ed.box == pytest.approx(GearBox(0.0, 0.0, 5.0, 2.0))
    _drag(ed, vp, (0.0, 0.0), (0.5, 0.25))  # esquina inferior izquierda
    assert ed.box == pytest.approx(GearBox(0.5, 0.25, 5.0, 2.0))


def test_the_box_does_not_jump_when_it_is_grabbed_off_center_of_the_handle() -> None:
    """Se arrastra por el desplazamiento, no hasta el puntero: agarrar el borde un poco adentro
    no lo hace saltar."""
    ed, vp, _ = _editor()
    grab = 4.0 - (HANDLE_PX / 2) / vp.scale  # adentro del borde, dentro de la tolerancia
    _drag(ed, vp, (grab, 1.0), (grab + 1.0, 1.0))
    assert ed.box == pytest.approx(GearBox(0.0, 0.0, 5.0, 2.0))


def test_resizing_has_a_minimum_size_in_pixels() -> None:
    ed, vp, _ = _editor()
    _drag(ed, vp, (4.0, 1.0), (-50.0, 1.0))
    assert ed.box is not None
    assert ed.box.width == pytest.approx(MIN_SIZE_PX / vp.scale)
    assert (ed.box.x0, ed.box.y0, ed.box.y1) == (0.0, 0.0, 2.0)


def test_a_click_on_the_box_changes_nothing() -> None:
    ed, vp, events = _editor()
    a = vp.world_to_screen(2.0, 1.0)
    assert ed.drag_begin(*a)
    ed.drag_end()  # sin ningún drag_update: es un clic
    assert ed.box == BOX and events == []


def test_a_drag_that_ends_where_it_started_changes_nothing() -> None:
    ed, vp, events = _editor()
    _drag(ed, vp, (2.0, 1.0), (2.0, 1.0))
    assert ed.box == BOX and events == []


def test_the_preview_is_shown_while_moving_and_cleared_afterwards() -> None:
    ed, vp, _ = _editor()
    previews: list[GearBox | None] = []
    ed.on_preview = lambda: previews.append(ed._preview)
    a, b = vp.world_to_screen(2.0, 1.0), vp.world_to_screen(3.0, 1.0)
    ed.drag_begin(*a)
    ed.drag_update(*b)
    assert previews[-1] == pytest.approx(BOX.translated(1.0, 0.0))
    assert ed.box == BOX  # la caja de verdad no cambia hasta soltar
    ed.drag_end()
    assert ed._preview is None and ed.box == pytest.approx(BOX.translated(1.0, 0.0))


def test_without_a_box_any_drag_draws_one() -> None:
    ed, vp, _ = _editor(None)
    _drag(ed, vp, (1.0, 1.0), (2.0, 2.0))
    assert ed.box == GearBox(1.0, 1.0, 2.0, 2.0)
