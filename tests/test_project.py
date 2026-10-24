import json
import math

import pytest

from mechchar.core.project import (
    FORMAT_VERSION,
    FitRecord,
    History,
    ProjectState,
    load_project,
    save_project,
)

SPLINE = tuple((3 * math.cos(t), 1.5 * math.sin(t)) for t in (0.0, 1.5, 3.0, 4.5))
FIT = FitRecord("four_bar", (1.0, 2.0, 3.0), branch=-1, ratio=2.0, rms_rel=0.012)


def full_state() -> ProjectState:
    return ProjectState(2, SPLINE, (-4.0, -3.0, 5.0, 6.0), FIT, (0.0, 0.5))


# --- estado y JSON -----------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    state = full_state()
    path = tmp_path / "p.json"
    save_project(path, state)
    assert load_project(path) == state
    assert json.loads(path.read_text())["version"] == FORMAT_VERSION
    assert not list(tmp_path.glob("*.tmp"))  # el archivo temporal no queda


def test_minimal_state_roundtrips(tmp_path):
    state = ProjectState(0, SPLINE)
    save_project(tmp_path / "m.json", state)
    loaded = load_project(tmp_path / "m.json")
    assert loaded == state and loaded.box is None and loaded.fit is None and loaded.phases == ()


def test_save_replaces_an_existing_file(tmp_path):
    path = tmp_path / "p.json"
    save_project(path, ProjectState(0, SPLINE))
    save_project(path, full_state())
    assert load_project(path) == full_state()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(version=99), "versión"),
        (lambda d: d.pop("scene"), "incompleto"),
        (lambda d: d.update(scene=-1), "escena"),
        (lambda d: d.update(spline=[[0, 0], [1, 1]]), "entre 3 y 256"),
        (lambda d: d.update(spline=[[0, 0], [1, 1], [2]]), "dos coordenadas"),
        (lambda d: d.update(spline=[[0, 0], [1, 1], [float("nan"), 2]]), "no finitos"),
        (lambda d: d.update(box=[0, 0, 1]), "4 valores"),
        (lambda d: d.update(box=[5, 0, 1, 1]), "mal formada"),
        (lambda d: d.update(phases=["a"]), "no son números"),
        (lambda d: d["fit"].update(branch=0), "rama"),
        (lambda d: d["fit"].update(params=[]), "parámetros"),
        (lambda d: d["fit"].update(params=[1, float("inf")]), "no finitos"),
        (lambda d: d["fit"].update(family=""), "familia"),
        (lambda d: d["fit"].update(rms_rel=-1), "inválidos"),
    ],
)
def test_invalid_projects_are_rejected_with_a_clear_message(mutate, match):
    d = full_state().to_dict()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        ProjectState.from_dict(d)


def test_files_that_are_not_projects_raise_value_error(tmp_path):
    for name, text in (("a.json", "no es json {"), ("b.json", "[1, 2]"), ("c.json", "null")):
        (tmp_path / name).write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_project(tmp_path / name)
    (tmp_path / "d.json").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="no es JSON"):
        load_project(tmp_path / "d.json")
    with pytest.raises(FileNotFoundError):  # un archivo que no existe no es un proyecto roto
        load_project(tmp_path / "nada.json")


# --- historial ---------------------------------------------------------------------------------


def test_undo_redo_walk_through_the_commits():
    h = History("a")
    assert not h.can_undo and not h.can_redo and h.undo() is None and h.redo() is None
    for s in "bcd":
        assert h.commit(s)
    assert h.present == "d"
    assert [h.undo(), h.undo(), h.undo(), h.undo()] == ["c", "b", "a", None]
    assert [h.redo(), h.redo(), h.redo(), h.redo()] == ["b", "c", "d", None]


def test_commit_of_an_unchanged_state_is_ignored():
    h = History("a")
    assert not h.commit("a") and not h.can_undo
    h.commit("b")
    assert not h.commit("b")
    assert h.undo() == "a"


def test_a_new_commit_after_undo_drops_the_redo_branch():
    h = History("a")
    h.commit("b")
    h.commit("c")
    h.undo()
    h.commit("x")
    assert not h.can_redo and h.present == "x"
    assert [h.undo(), h.undo()] == ["b", "a"]


def test_commits_with_the_same_key_close_in_time_collapse_into_one_step():
    h = History("a", coalesce_s=1.0)
    h.commit("b", "slider", now=10.0)
    h.commit("c", "slider", now=10.3)
    h.commit("d", "slider", now=10.6)
    assert h.present == "d"
    assert h.undo() == "a"  # un solo paso para todo el arrastre
    h.redo()
    h.commit("e", "slider", now=12.0)  # pasó más de 1 s: paso nuevo
    assert h.undo() == "d"
    h.redo()
    h.commit("f", "otro", now=12.1)  # otra clave: paso nuevo
    assert h.undo() == "e"


def test_the_first_commit_never_merges_into_the_initial_state():
    h = History("a")
    h.commit("b", "slider", now=0.0)
    assert h.undo() == "a"


def test_coalescing_does_not_swallow_the_redo_branch():
    h = History("a")
    h.commit("b", "k", now=0.0)
    h.commit("c", "k", now=0.1)
    h.undo()
    assert h.can_redo
    h.commit("d", "k", now=0.2)  # no pisa «b»: se descartan los rehacer y se apila
    assert not h.can_redo and h.undo() == "a"


def test_history_limit_drops_the_oldest_and_reset_starts_over():
    h = History(0, limit=5)
    for i in range(1, 20):
        h.commit(i)
    steps = 0
    while h.undo() is not None:
        steps += 1
    assert steps == 4 and h.present == 15  # quedan 5 fotos: 15..19
    h.reset("z")
    assert h.present == "z" and not h.can_undo and not h.can_redo
    with pytest.raises(ValueError):
        History(0, limit=1)


def test_history_works_with_project_states():
    h = History(ProjectState(0, SPLINE))
    moved = ProjectState(0, ((9.0, 9.0), *SPLINE[1:]))
    assert h.commit(moved)
    assert h.undo() == ProjectState(0, SPLINE)
    assert h.redo() == moved
