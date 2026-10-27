"""Tren de engranajes entre ejes (M5, núcleo): dientes enteros, paridad, cadenas y velocidades."""

import math

import pytest

from mechchar.core.geartrain import (
    MIN_TEETH,
    Gear,
    chain,
    chain_problems,
    connect,
    direction,
    distance,
    end_teeth_for_distance,
    intermediates_for,
    mesh_distance,
    meshes,
    overlaps,
    simulate_angles,
    speed_ratio,
)

M = 1.0
A = Gear(0.0, 0.0, 20, M)
B = Gear(20.0, 0.0, 20, M)  # a distancia 20 = m·z de dos engranajes de 20 dientes que engranan


# --- el engranaje ------------------------------------------------------------------------------


def test_radius_is_module_times_teeth_over_two() -> None:
    g = Gear(1.0, 2.0, 24, 0.5)
    assert g.radius == 6.0 and tuple(g.center) == (1.0, 2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(teeth=MIN_TEETH - 1, module=1.0),
        dict(teeth=20, module=0.0),
        dict(teeth=20, module=-1.0),
        dict(teeth=20, module=float("nan")),
    ],
)
def test_invalid_gears_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        Gear(0.0, 0.0, **kwargs)


def test_mesh_distance_is_module_times_half_the_sum_of_teeth() -> None:
    assert mesh_distance(Gear(0, 0, 20, 2.0), Gear(0, 0, 30, 2.0)) == 50.0


def test_gears_of_different_modules_cannot_be_compared() -> None:
    with pytest.raises(ValueError, match="módulo"):
        mesh_distance(Gear(0, 0, 20, 1.0), Gear(0, 0, 20, 2.0))
    with pytest.raises(ValueError, match="módulo"):
        connect(Gear(0, 0, 20, 1.0), Gear(20, 0, 20, 2.0), [])


def test_meshes_needs_the_exact_distance_and_the_same_layer() -> None:
    assert meshes(A, B)
    assert not meshes(A, Gear(21.0, 0.0, 20, M))  # separados
    assert not meshes(A, Gear(19.0, 0.0, 20, M))  # encimados
    assert not meshes(A, Gear(20.0, 0.0, 20, M, z=1.0))  # otra capa


# --- paridad -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "expected"), [(0, -1), (1, 1), (2, -1), (3, 1), (4, -1)])
def test_odd_number_of_intermediates_keeps_the_direction(k: int, expected: int) -> None:
    assert direction(k) == expected


def test_direction_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError):
        direction(-1)


@pytest.mark.parametrize(
    ("same", "at_least", "expected"),
    [(True, 0, 1), (False, 0, 0), (True, 2, 3), (False, 1, 2), (True, 1, 1), (False, 4, 4)],
)
def test_intermediates_for_gives_the_smallest_count_with_the_right_parity(
    same: bool, at_least: int, expected: int
) -> None:
    k = intermediates_for(same, at_least)
    assert k == expected and (direction(k) == 1) == same and k >= at_least


# --- dientes enteros ---------------------------------------------------------------------------


def test_end_teeth_for_distance_needs_an_integer_multiple_of_the_module() -> None:
    assert end_teeth_for_distance(20.0, 1.0) == 20
    assert end_teeth_for_distance(25.0, 2.5) == 10
    assert end_teeth_for_distance(20.5, 1.0) is None  # no es múltiplo
    assert end_teeth_for_distance(5.0, 1.0) is None  # menos de MIN_TEETH


# --- colocación de intermedios -----------------------------------------------------------------


def test_direct_mesh_only_works_at_exactly_the_mesh_distance() -> None:
    assert connect(A, B, []) == []
    assert connect(A, Gear(30.0, 0.0, 20, M), []) is None


def test_one_intermediate_sits_at_the_apex_of_the_triangle() -> None:
    (g,) = connect(A, B, [20]) or []
    assert distance(A, g) == pytest.approx(mesh_distance(A, g))
    assert distance(g, B) == pytest.approx(mesh_distance(g, B))
    assert (g.x, g.y) == pytest.approx((10.0, 10.0 * math.sqrt(3.0)))  # equilátero


def test_the_side_mirrors_the_chain() -> None:
    left, right = connect(A, B, [20], side=1), connect(A, B, [20], side=-1)
    assert left and right and left[0].y > 0 > right[0].y
    assert (left[0].x, left[0].y) == pytest.approx((right[0].x, -right[0].y))
    for inner in ([12, 14], [12, 12, 12]):
        up, down = connect(A, B, inner, side=1), connect(A, B, inner, side=-1)
        assert up and down
        assert [(g.x, g.y) for g in up] == pytest.approx([(g.x, -g.y) for g in down])


@pytest.mark.parametrize("inner", [[20], [10], [12, 14], [12, 12], [12, 12, 12], [10, 10, 10]])
def test_every_chain_found_meshes_without_overlaps(inner: list[int]) -> None:
    full = chain(A, B, inner)
    assert full is not None
    assert len(full) == len(inner) + 2 and full[0] == A and full[-1] == B
    assert chain_problems(full) == []
    for g, h in zip(full, full[1:], strict=False):
        assert distance(g, h) == pytest.approx(mesh_distance(g, h), rel=1e-9)


def test_the_straightest_chain_is_preferred() -> None:
    """Con un grado de libertad libre se devuelve la cadena más recta que no se pisa."""
    far = Gear(60.0, 0.0, 20, M)
    inner = connect(A, far, [20, 20]) or []
    assert len(inner) == 2
    assert all(abs(g.y) < 1e-9 for g in inner[:1])  # el primero queda sobre la recta
    # y es más recta que una cadena forzada a doblarse mucho
    bent = connect(A, B, [12, 14]) or []
    assert max(abs(g.y) for g in bent) > 5.0  # a 20 no hay lugar: se tuerce


def test_connect_returns_none_when_the_chain_cannot_close() -> None:
    assert connect(A, B, [10, 10, 10, 10]) is None  # límite documentado de la familia
    assert connect(A, Gear(200.0, 0.0, 20, M), [12]) is None  # demasiado lejos para un intermedio
    assert connect(A, B, [20, 20, 20, 20, 20, 20]) is None


def test_connect_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="capa"):
        connect(A, Gear(20.0, 0.0, 20, M, z=1.0), [20])
    with pytest.raises(ValueError, match="coinciden"):
        connect(A, Gear(0.0, 0.0, 20, M), [20])
    with pytest.raises(ValueError, match="dientes"):
        connect(A, B, [MIN_TEETH - 1])


# --- choques -----------------------------------------------------------------------------------


def test_overlaps_ignores_meshing_neighbours_and_other_layers() -> None:
    gears = [A, B, Gear(10.0, 0.0, 20, M)]  # el tercero pisa a los dos
    assert overlaps(gears) == [(0, 2), (1, 2)]
    assert overlaps(gears, ignore={frozenset((0, 2))}) == [(1, 2)]
    assert overlaps([A, B, Gear(10.0, 0.0, 20, M, z=2.0)]) == []  # otra capa no choca


def test_chain_problems_names_the_failure() -> None:
    assert chain_problems([A, Gear(25.0, 0.0, 20, M)]) == ["los engranajes 0 y 1 no engranan"]
    folded = [A, Gear(16.0, 0.0, 12, M), Gear(3.0, 0.0, 14, M), B]
    assert any("se pisan" in p for p in chain_problems(folded))


# --- velocidades por simulación ----------------------------------------------------------------


@pytest.mark.parametrize("inner", [[], [20], [12, 14], [12, 12, 12]])
def test_simulated_speeds_follow_the_parity_rule(inner: list[int]) -> None:
    full = chain(A, B, inner)
    assert full is not None
    angles = simulate_angles(full, 12)
    expected = direction(len(inner))  # extremos iguales: ±1
    assert angles[:, -1] == pytest.approx(expected * angles[:, 0])
    assert speed_ratio(full) == expected


def test_ends_with_different_teeth_change_the_speed_by_their_ratio() -> None:
    fast = Gear(16.0, 0.0, 12, M)  # engrana con A a 10 + 6
    full = chain(A, fast, [])
    assert full is not None
    angles = simulate_angles(full, 24)
    assert angles[:, 1] == pytest.approx(-20 / 12 * angles[:, 0])
    assert speed_ratio(full) == pytest.approx(-20 / 12)


def test_the_intermediates_do_not_change_the_speed_between_the_ends() -> None:
    ratios = []
    for t in (10, 14, 20, 31):
        full = chain(A, B, [t])
        assert full is not None
        angles = simulate_angles(full, 12)
        ratios.append(angles[5, -1] / angles[5, 0])
    assert ratios == pytest.approx([1.0] * 4)


def test_the_first_gear_makes_one_turn_per_cycle() -> None:
    full = chain(A, B, [12, 14])
    assert full is not None
    angles = simulate_angles(full, 24)
    assert angles[0, 0] == pytest.approx(0.0)
    assert angles[12, 0] == pytest.approx(math.pi)  # a mitad del ciclo, media vuelta
    # el intermedio de 12 dientes gira 20/12 de vuelta, en sentido contrario, por cada vuelta del
    # primero: a mitad del ciclo, 5/6 de vuelta (no un número entero)
    assert angles[12, 1] == pytest.approx(-20 / 12 * math.pi)


def test_simulate_refuses_a_train_that_does_not_mesh() -> None:
    with pytest.raises(ValueError, match="no engranan"):
        simulate_angles([A, Gear(25.0, 0.0, 20, M)])
    with pytest.raises(ValueError, match="al menos dos"):
        simulate_angles([A])
    with pytest.raises(ValueError, match="al menos dos"):
        speed_ratio([A])


# --- evitar otros engranajes -------------------------------------------------------------------


def test_connect_avoids_the_gears_it_is_told_to_avoid() -> None:
    """A distancia 34, con un engranaje de 10 dientes sobre el primer intermedio de la cadena
    [12, 14] más recta, hay otra cadena (más torcida) que lo esquiva (configuración medida)."""
    far = Gear(34.0, 0.0, 20, M)
    free = connect(A, far, [12, 14]) or []
    assert free
    blocker = Gear(free[0].x, free[0].y, 10, M)
    around = connect(A, far, [12, 14], avoid=[blocker])
    assert around is not None
    assert (around[0].x, around[0].y) != pytest.approx((free[0].x, free[0].y))
    full = [A, *around, far]
    assert chain_problems(full) == []
    assert overlaps([*full, blocker], ignore={frozenset((i, i + 1)) for i in range(3)}) == []


def test_a_blocker_that_leaves_no_feasible_angle_leaves_no_chain() -> None:
    """A distancia 42 el mismo bloqueador tapa todos los ángulos válidos (configuración medida)."""
    far = Gear(42.0, 0.0, 20, M)
    free = connect(A, far, [12, 14]) or []
    assert free
    assert connect(A, far, [12, 14], avoid=[Gear(free[0].x, free[0].y, 10, M)]) is None


def test_connect_gives_up_when_everything_is_blocked() -> None:
    wall = Gear(10.0, 5.0, 60, M)  # un engranaje enorme sobre la zona por la que pasa la cadena
    assert connect(A, B, [12, 14], avoid=[wall]) is None


def test_avoid_ignores_gears_in_other_layers() -> None:
    free = connect(A, B, [12, 14]) or []
    other_layer = Gear(free[0].x, free[0].y, 10, M, z=3.0)
    assert connect(A, B, [12, 14], avoid=[other_layer]) == free


def test_direct_mesh_also_respects_avoid() -> None:
    assert connect(A, B, [], avoid=[Gear(10.0, 0.0, 10, M)]) is None
    assert connect(A, B, [], avoid=[Gear(10.0, 40.0, 10, M)]) == []


def test_too_far_for_the_chain_is_rejected_without_searching() -> None:
    import time

    far = Gear(500.0, 0.0, 20, M)
    t0 = time.perf_counter()
    assert connect(A, far, [12, 14, 12]) is None
    assert time.perf_counter() - t0 < 0.05  # ni en línea recta llega: no se barren los 181 ángulos
