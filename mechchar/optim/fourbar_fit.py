"""Ajuste de un cuadrilátero articulado a una curva cerrada objetivo (M2).

Parámetros p = (ax, ay, phi, d, r, lc, rr, tx, ty, psi):
  (ax, ay, phi)  pose de la bancada en el mundo (pivote de la manivela y ángulo de la bancada)
  d, r, lc, rr   bancada, manivela, acoplador y balancín
  (tx, ty)       marcador, en coordenadas locales del acoplador
  psi            ángulo de la manivela (mundo) en t = 0, es decir el desfase del ciclo

Etapas: (1) los candidatos se generan al azar y se ordenan con una evaluación analítica vectorizada
(solo para elegir puntos de partida); (2) los mejores se refinan con el simulador genérico de
``core.assembly`` con el gradiente por el Teorema de la Función Implícita:
∂s/∂p = −(∂C/∂s)⁺ ∂C/∂p, con ∂C/∂s analítico y ∂C/∂p por diferencias finitas.

El objetivo no usa correspondencia punto a punto: el marcador no recorre su curva a velocidad
uniforme, así que exigir x(p, k/n) = x̂_k deja un error que ningún optimizador puede bajar (el
paper también descarta el timing en la métrica). Se minimizan distancias punto-a-curva en los dos
sentidos (mecanismo → objetivo y objetivo → mecanismo), con residuo escalar a lo largo de la
normal del segmento más cercano, lo que deja libre el deslizamiento tangencial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from mechchar.core.assembly import Assembly, Drive, FixedState, Pin, drot, world_point
from mechchar.core.mechanisms import circle_intersections
from mechchar.optim.curves import bbox_diag, chamfer_rms, resample_closed

N_PARAMS = 10
TWO_PI = 2.0 * math.pi


@dataclass
class Built:
    asm: Assembly
    state0: np.ndarray
    coupler: int
    tracer: tuple[float, float]


def build(p: np.ndarray, branch: int) -> Built | None:
    """Ensamblaje para los parámetros ``p``; None si no se puede armar en t = 0."""
    ax, ay, phi, d, r, lc, rr, tx, ty, psi = (float(v) for v in p)
    if min(d, r, lc, rr) <= 0.0:
        return None
    th_local = psi - phi
    b_loc = np.array([r * math.cos(th_local), r * math.sin(th_local)])
    d_loc = np.array([d, 0.0])
    try:
        cands = circle_intersections(tuple(b_loc), lc, (d, 0.0), rr)
    except ValueError:
        return None
    p_loc = cands[0] if branch > 0 else cands[1]
    c, s = math.cos(phi), math.sin(phi)
    rot_g = np.array([[c, -s], [s, c]])
    origin = np.array([ax, ay])
    b_w, d_w = origin + rot_g @ b_loc, origin + rot_g @ d_loc

    asm = Assembly()
    g = asm.add_component("ground", ax, ay, phi)
    cr = asm.add_component("crank", ax, ay, psi)
    k = asm.add_component(
        "coupler", b_w[0], b_w[1], phi + math.atan2(p_loc[1] - b_loc[1], p_loc[0] - b_loc[0])
    )
    ro = asm.add_component(
        "rocker", d_w[0], d_w[1], phi + math.atan2(p_loc[1] - d_loc[1], p_loc[0] - d_loc[0])
    )
    asm.add(FixedState(g, ax, ay, phi))
    asm.add(Pin(g, (0.0, 0.0), cr, (0.0, 0.0)))
    asm.add(Pin(cr, (r, 0.0), k, (0.0, 0.0)))
    asm.add(Pin(k, (lc, 0.0), ro, (rr, 0.0)))
    asm.add(Pin(ro, (0.0, 0.0), g, (d, 0.0)))
    asm.add(Drive(cr, offset=psi))
    return Built(asm, asm.initial_state(), k, (tx, ty))


def trace_curve(p: np.ndarray, branch: int, n: int) -> np.ndarray | None:
    """Curva del marcador en n fases t = k/n, o None si el mecanismo no completa el ciclo."""
    b = build(p, branch)
    if b is None:
        return None
    res = b.asm.trace_cycle(b.state0, n_steps=n, max_jump=1.0)
    if not res.ok:
        return None
    return np.array([world_point(s, b.coupler, b.tracer) for s in res.states])


def curve_and_jacobian(
    p: np.ndarray, branch: int, n: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Curva x(p) (n, 2) y su Jacobiano dx/dp (n, 2, N_PARAMS), por función implícita."""
    base = build(p, branch)
    if base is None:
        return None
    res = base.asm.trace_cycle(base.state0, n_steps=n, max_jump=1.0)
    if not res.ok:
        return None

    steps = 1e-6 * (1.0 + np.abs(p))
    plus: list[Built] = []
    minus: list[Built] = []
    for j in range(N_PARAMS):
        pp, pm = p.copy(), p.copy()
        pp[j] += steps[j]
        pm[j] -= steps[j]
        bp, bm = build(pp, branch), build(pm, branch)
        if bp is None or bm is None:
            return None
        plus.append(bp)
        minus.append(bm)

    x = np.zeros((n, 2))
    dx = np.zeros((n, 2, N_PARAMS))
    i = 3 * base.coupler
    for k in range(n):
        t, s = k / n, res.states[k]
        jac_s = base.asm.jacobian(s, t)
        c_p = np.empty((jac_s.shape[0], N_PARAMS))
        e_p = np.empty((2, N_PARAMS))
        for j in range(N_PARAMS):
            c_p[:, j] = (plus[j].asm.residual(s, t) - minus[j].asm.residual(s, t)) / (2 * steps[j])
            e_p[:, j] = (
                world_point(s, base.coupler, plus[j].tracer)
                - world_point(s, base.coupler, minus[j].tracer)
            ) / (2 * steps[j])
        ds_dp = -np.linalg.lstsq(jac_s, c_p, rcond=None)[0]
        e_s = np.zeros((2, jac_s.shape[1]))
        e_s[0, i] = 1.0
        e_s[1, i + 1] = 1.0
        e_s[:, i + 2] = drot(s[i + 2]) @ np.asarray(base.tracer)
        x[k] = world_point(s, base.coupler, base.tracer)
        dx[k] = e_s @ ds_dp + e_p
    return x, dx


# --- evaluación analítica vectorizada (solo para elegir candidatos) -----------------------------


def _fast_local_curves(
    r: np.ndarray, lc: np.ndarray, rr: np.ndarray, tx: np.ndarray, ty: np.ndarray,
    branch: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Curvas del marcador en el marco de la bancada (d = 1, manivela en ángulo 0 en t = 0).

    Devuelve (curvas (K, n, 2), válidas (K,)); válida = el ciclo completo es ensamblable."""
    th = TWO_PI * np.arange(n) / n
    bx, by = r[:, None] * np.cos(th)[None, :], r[:, None] * np.sin(th)[None, :]
    vx, vy = 1.0 - bx, -by
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.hypot(vx, vy)
        a = (lc[:, None] ** 2 - rr[:, None] ** 2 + dist**2) / (2.0 * dist)
        h2 = lc[:, None] ** 2 - a**2
        h = np.sqrt(np.maximum(h2, 0.0))
        ux, uy = vx / dist, vy / dist
    valid = np.isfinite(h2).all(1) & (h2 > (0.02 * lc[:, None]) ** 2).all(1)
    px = bx + a * ux - branch[:, None] * h * uy
    py = by + a * uy + branch[:, None] * h * ux
    ang = np.arctan2(py - by, px - bx)
    c, s = np.cos(ang), np.sin(ang)
    qx = bx + c * tx[:, None] - s * ty[:, None]
    qy = by + s * tx[:, None] + c * ty[:, None]
    return np.stack([qx, qy], axis=-1), valid


def _principal_angle(curves: np.ndarray) -> np.ndarray:
    """Ángulo del eje principal de cada curva (K, n, 2) -> (K,)."""
    d = curves - curves.mean(1, keepdims=True)
    cxx, cyy = (d[..., 0] ** 2).sum(1), (d[..., 1] ** 2).sum(1)
    cxy = (d[..., 0] * d[..., 1]).sum(1)
    return 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)


@dataclass
class FitResult:
    params: np.ndarray
    branch: int
    rms: float  # distancia chamfer RMS entre la curva final y el objetivo
    rms_rel: float  # rms / diagonal de la caja del objetivo
    curve: np.ndarray  # curva final, densa (n_dense, 2)
    initial_rms_rel: float  # error del mejor candidato antes del refinamiento continuo


def _initial_candidates(
    y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, int, float]]:
    """Mejores candidatos aleatorios: (p0, branch, error relativo)."""
    center, size = y.mean(0), bbox_diag(y)
    ang_y = float(_principal_angle(y[None])[0])
    pool: list[tuple[float, np.ndarray, int, np.ndarray]] = []
    chunk = 4000
    for _ in range(max(1, n_candidates // chunk)):
        r = rng.uniform(0.1, 1.0, chunk)
        lc = rng.uniform(0.6, 3.0, chunk)
        rr = rng.uniform(0.6, 3.0, chunk)
        tx = lc * rng.uniform(-0.5, 1.5, chunk)
        ty = lc * rng.uniform(-1.0, 1.0, chunk)
        br = rng.choice([-1, 1], chunk)
        local, valid = _fast_local_curves(r, lc, rr, tx, ty, br.astype(float), n)
        keep = np.where(valid)[0]
        if keep.size == 0:
            continue
        local, r, lc, rr, tx, ty, br = (a[keep] for a in (local, r, lc, rr, tx, ty, br))
        c0 = local.mean(1, keepdims=True)
        diag = np.hypot(np.ptp(local[..., 0], axis=1), np.ptp(local[..., 1], axis=1))
        f = size / diag
        ang_l = _principal_angle(local)
        for flip in (0.0, math.pi):
            phi = ang_y - ang_l + flip
            cph, sph = np.cos(phi)[:, None], np.sin(phi)[:, None]
            dxy = (local - c0) * f[:, None, None]
            pts = np.stack(
                [cph * dxy[..., 0] - sph * dxy[..., 1], sph * dxy[..., 0] + cph * dxy[..., 1]],
                axis=-1,
            ) + center
            d2 = ((pts[:, :, None, :] - y[None, None, :, :]) ** 2).sum(-1)
            cost = np.sqrt(0.5 * (d2.min(2).mean(1) + d2.min(1).mean(1)))
            for idx in np.argsort(cost)[:n_keep]:
                fc = f[idx] * c0[idx, 0]  # centroide local escalado
                c_, s_ = cph[idx, 0], sph[idx, 0]
                a_world = center - np.array([c_ * fc[0] - s_ * fc[1], s_ * fc[0] + c_ * fc[1]])
                p0 = np.array(
                    [a_world[0], a_world[1], phi[idx], f[idx], f[idx] * r[idx], f[idx] * lc[idx],
                     f[idx] * rr[idx], f[idx] * tx[idx], f[idx] * ty[idx], phi[idx]]
                )
                pool.append((float(cost[idx]), p0, int(br[idx]), pts[idx]))
    pool.sort(key=lambda item: item[0])
    return [(p0, br, cost / size) for cost, p0, br, _ in pool[:n_keep]]


def _closest_on_polyline(
    q: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segmento más cercano de la poligonal cerrada ``v`` (M, 2) para cada punto de ``q`` (m, 2).

    Devuelve (índice del segmento, parámetro λ ∈ [0, 1] del pie, normal unitaria del segmento)."""
    a, ab = v, np.roll(v, -1, axis=0) - v
    ab2 = np.maximum((ab**2).sum(1), 1e-18)
    lam = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)
    foot = a[None] + lam[..., None] * ab[None]
    idx = ((q[:, None, :] - foot) ** 2).sum(-1).argmin(1)
    rows = np.arange(len(q))
    seg = ab[idx]
    norm = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-12)
    normal = np.column_stack([-seg[:, 1], seg[:, 0]]) / norm[:, None]
    return idx, lam[rows, idx], normal


def distance_residuals(
    x: np.ndarray, dx: np.ndarray, y: np.ndarray, y_dense: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Residuos y Jacobiano del objetivo sin correspondencia (2n residuos).

    x (n, 2) curva del mecanismo, dx (n, 2, P) su Jacobiano, y (m, 2) muestras del objetivo y
    y_dense (M, 2) su poligonal densa."""
    n, n_params = len(x), dx.shape[2]
    # mecanismo -> objetivo
    idx_f, lam_f, nrm_f = _closest_on_polyline(x, y_dense)
    foot = y_dense[idx_f] + lam_f[:, None] * (np.roll(y_dense, -1, axis=0)[idx_f] - y_dense[idx_f])
    r_f = (nrm_f * (x - foot)).sum(1)
    j_f = np.einsum("ki,kip->kp", nrm_f, dx)
    # objetivo -> mecanismo
    idx_b, lam_b, nrm_b = _closest_on_polyline(y, x)
    nxt = (idx_b + 1) % n
    foot_b = (1 - lam_b)[:, None] * x[idx_b] + lam_b[:, None] * x[nxt]
    r_b = (nrm_b * (y - foot_b)).sum(1)
    d_foot = (1 - lam_b)[:, None, None] * dx[idx_b] + lam_b[:, None, None] * dx[nxt]
    j_b = -np.einsum("mi,mip->mp", nrm_b, d_foot)
    # mismo número de residuos para ambos sentidos: n del mecanismo y len(y) del objetivo
    return np.concatenate([r_f, r_b]), np.vstack([j_f, j_b]).reshape(-1, n_params)


def _refine(
    p0: np.ndarray, branch: int, y: np.ndarray, y_dense: np.ndarray, n: int, max_nfev: int
) -> np.ndarray:
    size = bbox_diag(y)
    n_res = n + len(y)
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = p.tobytes()
        if key not in cache:
            cache.clear()
            out = curve_and_jacobian(p, branch, n)
            if out is None:  # mecanismo no ensamblable: penalización para rechazar el paso
                cache[key] = (np.full(n_res, size), np.zeros((n_res, N_PARAMS)))
            else:
                cache[key] = distance_residuals(out[0], out[1], y, y_dense)
        return cache[key]

    res = least_squares(
        lambda p: evaluate(p)[0],
        p0,
        jac=lambda p: evaluate(p)[1],
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    return np.asarray(res.x)


def fit_four_bar(
    target: np.ndarray,
    *,
    n: int = 48,
    n_candidates: int = 16000,
    n_refine: int = 2,
    max_nfev: int = 25,
    n_dense: int = 180,
    seed: int = 0,
) -> FitResult | None:
    """Ajusta un cuadrilátero a ``target`` (curva cerrada (m, 2)); None si no hay ninguno válido."""
    y = resample_closed(target, n)
    y_dense = resample_closed(target, n_dense)
    size = bbox_diag(y)
    rng = np.random.default_rng(seed)
    starts = _initial_candidates(y, n, n_candidates, n_refine, rng)
    best: FitResult | None = None
    for p0, branch, init_rel in starts:
        p = _refine(p0, branch, y, y_dense, n, max_nfev)
        curve = trace_curve(p, branch, n_dense)
        if curve is None:
            continue
        rms = chamfer_rms(curve, y_dense)
        if best is None or rms < best.rms:
            best = FitResult(p, branch, rms, rms / size, curve, init_rel)
    return best
