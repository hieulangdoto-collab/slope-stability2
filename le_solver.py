"""
Slope Stability Limit Equilibrium (LE) Solver — 2D circular / composite slip surfaces.

Implements the classical General Limit Equilibrium (GLE) framework:
    - Fellenius / Ordinary method    (no interslice forces)
    - Bishop Simplified method       (interslice normal forces only, moment eq.)
    - Spencer method                 (f(x) = 1 constant; both force & moment eq.)
    - Morgenstern-Price method       (f(x) = half-sine varying along the slip
                                       surface; both force & moment eq.)

v3 changes vs v2:
    - Composite slip surfaces: a circular arc that is constrained to follow a
      designated weak-layer interface wherever the circle would otherwise cut
      deeper into stronger material below it.
    - External loads: distributed surcharge and vertical point loads.
    - Reinforcement: horizontal tensile force(s) crossing the slip surface,
      added as a resisting contribution to both moment and force equilibrium.

Still-simplified assumptions (documented; see README for extension ideas):
    - Mohr-Coulomb strength per layer (no Hoek-Brown / SHANSEP / anisotropic yet).
    - Optional single piezometric surface (one water table for the whole slope).
    - Reinforcement force is assumed to act HORIZONTALLY (typical simplification
      for soil nails / geogrid in basic LE formulations); inclined/vertical
      reinforcement components are not modeled.
    - External loads are VERTICAL only (surcharge / point load); no horizontal
      seismic or applied shear loads yet.
"""

import math
import numpy as np

GAMMA_W = 9.81  # kN/m3


# ---------------------------------------------------------------------------
# Materials & stratigraphy
# ---------------------------------------------------------------------------

class Material:
    """Mohr-Coulomb material: c (kPa), phi (deg), unit_weight (kN/m3)."""

    def __init__(self, name, c, phi_deg, unit_weight):
        self.name = name
        self.c = c
        self.phi = math.radians(phi_deg)
        self.unit_weight = unit_weight

    def __repr__(self):
        return f"Material({self.name}: c={self.c}, phi={math.degrees(self.phi):.1f}deg, gamma={self.unit_weight})"


class Layer:
    """One stratigraphic layer. `lower_boundary` is a polyline [(x,y),...] marking
    the BOTTOM of this layer (i.e. the top of the next layer down). Use
    `lower_boundary=None` for the last (deepest) layer, which then extends to
    -infinity (acts as a very thick / bedrock-like final unit)."""

    def __init__(self, material, lower_boundary=None):
        self.material = material
        self.lower_boundary = lower_boundary


# ---------------------------------------------------------------------------
# External loads & reinforcement
# ---------------------------------------------------------------------------

class Surcharge:
    """Uniformly distributed vertical load q (kPa) applied to the ground
    surface between x_start and x_end."""

    def __init__(self, x_start, x_end, q):
        self.x_start = x_start
        self.x_end = x_end
        self.q = q


class PointLoad:
    """Vertical point load P (kN per m out-of-plane) applied at x."""

    def __init__(self, x, P):
        self.x = x
        self.P = P


class Reinforcement:
    """Horizontal tensile reinforcement force T (kN per m out-of-plane) crossing
    the slip surface at horizontal location x (e.g. a soil nail, geogrid layer,
    or anchor). T should already reflect the governing (minimum of tensile,
    pullout, anchorage) capacity -- this solver does not check capacity limits.

    Simplification: force assumed horizontal (see module docstring)."""

    def __init__(self, x, T):
        self.x = x
        self.T = T


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def interp_polyline(pts, x):
    """Linearly interpolate y at coordinate x along a polyline pts=[(x0,y0),...] sorted by x."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def circle_intersections(ground_pts, xc, yc, R):
    """Find the left/right x where the circle (xc,yc,R) intersects the ground surface polyline.
    Returns (x_left, x_right) or None if no valid two-point intersection is found."""
    xs = np.linspace(ground_pts[0][0], ground_pts[-1][0], 4000)
    ys = np.array([interp_polyline(ground_pts, x) for x in xs])
    under = R ** 2 - (xs - xc) ** 2
    circle_y_lower = np.where(under >= 0, yc - np.sqrt(np.where(under >= 0, under, 0.0)), np.nan)
    diff = ys - circle_y_lower
    valid = ~np.isnan(diff)
    if valid.sum() < 2:
        return None
    xs_v = xs[valid]
    diff_v = diff[valid]
    crossings = []
    for i in range(len(xs_v) - 1):
        if diff_v[i] == 0:
            crossings.append(xs_v[i])
        elif diff_v[i] * diff_v[i + 1] < 0:
            t = diff_v[i] / (diff_v[i] - diff_v[i + 1])
            crossings.append(xs_v[i] + t * (xs_v[i + 1] - xs_v[i]))
    if len(crossings) < 2:
        return None
    return min(crossings), max(crossings)


def _layer_bands_at_x(x, ground_pts, layers):
    """Returns list of (material, upper_y, lower_y) for each layer at this x,
    from the ground surface down. The last layer's lower_y is -1e9 (proxy -inf)."""
    bands = []
    upper = interp_polyline(ground_pts, x)
    for layer in layers:
        if layer.lower_boundary is None:
            lower = -1.0e9
        else:
            lower = interp_polyline(layer.lower_boundary, x)
        bands.append((layer.material, upper, lower))
        upper = lower
    return bands


def _column_weight_and_base_material(x, y_top, y_base, width, ground_pts, layers):
    """Integrate unit weight through however many layers a column crosses, and
    identify which layer's material governs strength at the column base."""
    bands = _layer_bands_at_x(x, ground_pts, layers)
    W = 0.0
    base_material = layers[-1].material  # fallback
    for material, upper, lower in bands:
        seg_top = min(upper, y_top)
        seg_bot = max(lower, y_base)
        if seg_top > seg_bot:
            W += material.unit_weight * (seg_top - seg_bot) * width
        if lower <= y_base <= upper or (lower <= y_base and upper >= y_base):
            base_material = material
    return W, base_material


def _add_surface_loads(x, width, W, surcharges, point_loads):
    """Add vertical surcharge / point-load contributions to a column's weight W."""
    if surcharges:
        x0, x1 = x - width / 2.0, x + width / 2.0
        for sc in surcharges:
            ov = min(x1, sc.x_end) - max(x0, sc.x_start)
            if ov > 0:
                W += sc.q * ov
    if point_loads:
        x0, x1 = x - width / 2.0, x + width / 2.0
        for pl in point_loads:
            if x0 <= pl.x < x1:
                W += pl.P
    return W


# ---------------------------------------------------------------------------
# Slice generation (circular and composite)
# ---------------------------------------------------------------------------

class Slice:
    __slots__ = ("x_mid", "width", "y_top", "y_base", "alpha", "W", "l", "u", "c", "phi", "on_weak")

    def __init__(self, x_mid, width, y_top, y_base, alpha, W, l, u, c, phi, on_weak=False):
        self.x_mid = x_mid
        self.width = width
        self.y_top = y_top
        self.y_base = y_base
        self.alpha = alpha   # base inclination, radians. Positive => "driving" side.
        self.W = W           # column weight (kN per m out-of-plane), incl. surface loads
        self.l = l           # base length = width / cos(alpha)
        self.u = u           # pore water pressure at base midpoint (kPa)
        self.c = c           # cohesion of the material AT THE BASE (kPa)
        self.phi = phi       # friction angle of the material AT THE BASE (radians)
        self.on_weak = on_weak  # True if this slice's base was clipped onto a weak layer


def _make_slice(xm, width, y_top, y_base, alpha, ground_pts, layers, piezo_pts,
                 surcharges, point_loads, on_weak=False):
    W, base_material = _column_weight_and_base_material(xm, y_top, y_base, width, ground_pts, layers)
    W = _add_surface_loads(xm, width, W, surcharges, point_loads)
    l = width / math.cos(alpha)
    u = 0.0
    if piezo_pts is not None:
        yw = interp_polyline(piezo_pts, xm)
        head = yw - y_base
        if head > 0:
            u = GAMMA_W * head
    return Slice(xm, width, y_top, y_base, alpha, W, l, u, base_material.c, base_material.phi, on_weak)


def generate_slices(ground_pts, xc, yc, R, n_slices, layers, piezo_pts=None,
                     surcharges=None, point_loads=None, weak_boundary=None):
    """Discretize the slip surface between its ground-surface intersections into
    n_slices vertical columns.

    ground_pts, piezo_pts: polylines [(x,y),...]
    layers: list of Layer objects, ordered top to bottom.
    weak_boundary: OPTIONAL. If given, produces a COMPOSITE surface: wherever
        the circular arc would cut below this boundary, the surface is instead
        clipped to follow along the boundary (representing a slip surface that
        "seeks out" and rides along a weak interface rather than cutting into
        stronger material beneath it).
        Accepts either a single polyline, OR a list of polylines (multiple
        nested weak interfaces) -- at each x, the surface follows whichever
        weak boundary is SHALLOWEST among those the circle would otherwise cut
        below (the kinematically preferred, least-resistance path).
    """
    inter = circle_intersections(ground_pts, xc, yc, R)
    if inter is None:
        return None
    x_left, x_right = inter
    if x_right - x_left < 1e-6:
        return None

    weak_list = None
    if weak_boundary is not None:
        # normalize to a list of polylines: a polyline is a list of (x,y) points,
        # so check whether the first element is itself a point or a polyline.
        first_elem = weak_boundary[0]
        if isinstance(first_elem[0], (int, float)):
            weak_list = [weak_boundary]  # a single polyline was passed
        else:
            weak_list = list(weak_boundary)  # already a list of polylines

    edges = np.linspace(x_left, x_right, n_slices + 1)
    slices = []
    for i in range(n_slices):
        x0, x1 = edges[i], edges[i + 1]
        xm = 0.5 * (x0 + x1)
        width = x1 - x0
        y_top = interp_polyline(ground_pts, xm)
        under_sqrt = R ** 2 - (xm - xc) ** 2
        if under_sqrt <= 0:
            return None
        y_base_circle = yc - math.sqrt(under_sqrt)
        if y_base_circle >= y_top:
            return None

        on_weak = False
        active_boundary = None
        if weak_list is not None:
            best_y = y_base_circle
            for wb in weak_list:
                y_weak = min(interp_polyline(wb, xm), y_top - 1e-6)
                # Ignore boundaries that have pinched out to (near) the ground
                # surface here -- sliding "along" a layer with ~zero cover is
                # degenerate (zero driving weight) and not a meaningful failure
                # path; skip it rather than let it truncate the slide early.
                MIN_COVER = 0.05
                if y_top - y_weak < MIN_COVER:
                    continue
                if y_weak > best_y:
                    best_y = y_weak
                    active_boundary = wb
            if active_boundary is not None:
                y_base = best_y
                on_weak = True
            else:
                y_base = y_base_circle
        else:
            y_base = y_base_circle

        if y_base >= y_top:
            return None

        if on_weak:
            # local slope of the active weak boundary via central finite difference
            dx = max(width * 0.1, 1e-3)
            y_l = interp_polyline(active_boundary, xm - dx)
            y_r = interp_polyline(active_boundary, xm + dx)
            dydx = (y_r - y_l) / (2 * dx)
            alpha = math.atan(-dydx)
        else:
            dydx = (xm - xc) / math.sqrt(under_sqrt)
            alpha = math.atan(-dydx)

        slices.append(_make_slice(xm, width, y_top, y_base, alpha, ground_pts, layers,
                                   piezo_pts, surcharges, point_loads, on_weak))
    return slices


# ---------------------------------------------------------------------------
# Reinforcement bookkeeping
# ---------------------------------------------------------------------------

def _reinforcement_terms(slices, xc, yc, R, reinforcements):
    """Returns (moment_denominator_reduction, force_denominator_reduction).
    Each reinforcement's horizontal force T is assigned to the slice whose
    base is closest to its x-location. See derivation in README ("Reinforcement")."""
    if not reinforcements:
        return 0.0, 0.0
    moment_term = 0.0
    force_term = 0.0
    for r in reinforcements:
        nearest = min(slices, key=lambda s: abs(s.x_mid - r.x))
        moment_term += r.T * (yc - nearest.y_base)
        force_term += r.T
    return moment_term / R, force_term


# ---------------------------------------------------------------------------
# Limit equilibrium methods
# ---------------------------------------------------------------------------

def fellenius_fs(slices, kh=0.0, kv=0.0, xc=None, yc=None, R=None, moment_reduction=0.0):
    """Ordinary (Fellenius) method — closed form, no interslice forces, no iteration.
    kh, kv: pseudo-static seismic coefficients (horizontal, vertical). If kh != 0,
    xc, yc, R must be given (needed for the seismic force's moment arm)."""
    num = 0.0
    den = 0.0
    for s in slices:
        W_eff = (1.0 + kv) * s.W
        Nf = W_eff * math.cos(s.alpha) - s.u * s.l
        num += s.c * s.l + max(Nf, 0.0) * math.tan(s.phi)
        den += W_eff * math.sin(s.alpha)
        if kh != 0.0 and R:
            y_mid = 0.5 * (s.y_top + s.y_base)
            den += kh * s.W * (yc - y_mid) / R
    den -= moment_reduction
    if den <= 0:
        return None
    return num / den


def _N_i(s, F, lam, fx, kh=0.0, kv=0.0):
    a = s.alpha
    t = lam * fx
    p = math.cos(a) + t * math.sin(a)
    q = math.sin(a) - t * math.cos(a)
    denom = p + (math.tan(s.phi) / F) * q
    if abs(denom) < 1e-9:
        return None
    W_eff = (1.0 + kv) * s.W
    numer = W_eff + t * kh * s.W - (q / F) * (s.c * s.l - s.u * s.l * math.tan(s.phi))
    return numer / denom


def _bisect_F(calc_F, F_guess=1.2, iters=100, tol=1e-6):
    """Solve F = calc_F(F) by damped fixed-point iteration (F_new = calc_F(F),
    then step halfway there). This is the same scheme used since v1-v4 and is
    reliable for the vast majority of cases; it is NOT a bulletproof global
    solver. See README 'Loi da phat hien va sua' (v5) for a documented case
    (high seismic kh combined with large lambda) where this can converge to
    a different branch depending on F_guess -- for production use, a more
    robust bounded-bisection or 2D Newton solver on (F, lambda) jointly would
    be a worthwhile upgrade."""
    F = F_guess
    for _ in range(iters):
        F_new = calc_F(F)
        if F_new is None:
            return None
        if abs(F_new - F) < tol:
            return F_new
        F = 0.5 * F + 0.5 * F_new
    return F


def _solve_Fm(slices, lam, fx_list, F_guess=1.2, moment_reduction=0.0,
              kh=0.0, kv=0.0, xc=None, yc=None, R=None):
    def calc_F(F):
        num = 0.0
        den = 0.0
        for s, fx in zip(slices, fx_list):
            N = _N_i(s, F, lam, fx, kh=kh, kv=kv)
            if N is None:
                return None
            num += s.c * s.l + (N - s.u * s.l) * math.tan(s.phi)
            den += (1.0 + kv) * s.W * math.sin(s.alpha)
            if kh != 0.0 and R:
                y_mid = 0.5 * (s.y_top + s.y_base)
                den += kh * s.W * (yc - y_mid) / R
        den -= moment_reduction
        if den <= 0:
            return None
        return num / den
    return _bisect_F(calc_F, F_guess=F_guess)


def _solve_Ff(slices, lam, fx_list, F_guess=1.2, force_reduction=0.0,
              kh=0.0, kv=0.0):
    def calc_F(F):
        num = 0.0
        den = 0.0
        for s, fx in zip(slices, fx_list):
            N = _N_i(s, F, lam, fx, kh=kh, kv=kv)
            if N is None:
                return None
            num += (s.c * s.l + (N - s.u * s.l) * math.tan(s.phi)) * math.cos(s.alpha)
            den += N * math.sin(s.alpha)
            if kh != 0.0:
                den += kh * s.W
        den -= force_reduction
        if den <= 0:
            return None
        return num / den
    return _bisect_F(calc_F, F_guess=F_guess)


def bishop_fs(slices, F_guess=1.2, xc=None, yc=None, R=None, reinforcements=None, kh=0.0, kv=0.0):
    """Bishop Simplified method: moment equilibrium only, interslice shear = 0 (lam=0).
    Pass xc, yc, R and `reinforcements` to include reinforcement resisting moment.
    kh, kv: pseudo-static seismic coefficients (need xc,yc,R if kh != 0)."""
    fx_list = [1.0] * len(slices)
    moment_reduction = 0.0
    if reinforcements:
        moment_reduction, _ = _reinforcement_terms(slices, xc, yc, R, reinforcements)
    return _solve_Fm(slices, lam=0.0, fx_list=fx_list, F_guess=F_guess, moment_reduction=moment_reduction,
                      kh=kh, kv=kv, xc=xc, yc=yc, R=R)


def half_sine_fx(slices):
    """Morgenstern-Price interslice-force function f(x): half-sine shape, zero at
    the entry and exit of the slip surface, peak at the middle."""
    xs = [s.x_mid for s in slices]
    x_min, x_max = min(xs), max(xs)
    span = max(x_max - x_min, 1e-9)
    return [math.sin(math.pi * (x - x_min) / span) for x in xs]


def _gle_fs(slices, fx_list, F_guess=1.2, lam_scan_range=(-1.2, 1.2), lam_scan_n=81,
            lam_tol=1e-6, max_bisect=60, moment_reduction=0.0, force_reduction=0.0,
            kh=0.0, kv=0.0, xc=None, yc=None, R=None):
    """Shared GLE solver (Spencer / Morgenstern-Price): finds lam such that Fm=Ff.
    See le_solver v2 notes: picks the bracket closest to lam=0 to avoid spurious
    far-from-zero roots caused by denominator near-singularities."""

    def gap(lam):
        Fm = _solve_Fm(slices, lam, fx_list, F_guess=F_guess, moment_reduction=moment_reduction,
                        kh=kh, kv=kv, xc=xc, yc=yc, R=R)
        Ff = _solve_Ff(slices, lam, fx_list, F_guess=F_guess, force_reduction=force_reduction,
                        kh=kh, kv=kv)
        if Fm is None or Ff is None:
            return None, None, None
        return Fm - Ff, Fm, Ff

    lo_range, hi_range = lam_scan_range
    lams = np.linspace(lo_range, hi_range, lam_scan_n)
    gaps = [gap(l)[0] for l in lams]

    brackets = []
    for i in range(len(lams) - 1):
        if gaps[i] is not None and gaps[i + 1] is not None and gaps[i] * gaps[i + 1] < 0:
            brackets.append((lams[i], lams[i + 1]))
    if not brackets:
        # Fallback: for some composite/degenerate-slice geometries, Fm(lam) and
        # Ff(lam) can approach each other very closely without ever crossing
        # sign (a near-tangent minimum of |gap|). If the closest approach is
        # small, accept it as an approximate converged solution rather than
        # failing outright.
        valid = [(l, g) for l, g in zip(lams, gaps) if g is not None]
        if not valid:
            return None
        best_lam, best_gap = min(valid, key=lambda t: abs(t[1]))
        if abs(best_gap) < 0.02:
            Fm = _solve_Fm(slices, best_lam, fx_list, F_guess=F_guess, moment_reduction=moment_reduction,
                            kh=kh, kv=kv, xc=xc, yc=yc, R=R)
            Ff = _solve_Ff(slices, best_lam, fx_list, F_guess=F_guess, force_reduction=force_reduction,
                            kh=kh, kv=kv)
            if Fm is not None and Ff is not None:
                return 0.5 * (Fm + Ff)
        return None
    lo, hi = min(brackets, key=lambda b: abs(0.5 * (b[0] + b[1])))
    g_lo, _, _ = gap(lo)

    Fm_mid = Ff_mid = None
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        g_mid, Fm_mid, Ff_mid = gap(mid)
        if g_mid is None:
            return None
        if abs(g_mid) < lam_tol:
            return 0.5 * (Fm_mid + Ff_mid)
        if g_lo * g_mid < 0:
            hi = mid
        else:
            lo = mid
            g_lo = g_mid
    return 0.5 * (Fm_mid + Ff_mid) if Fm_mid is not None else None


def spencer_fs(slices, F_guess=1.2, xc=None, yc=None, R=None, reinforcements=None, kh=0.0, kv=0.0):
    """Spencer method: f(x) = 1.0 (constant interslice force angle)."""
    fx_list = [1.0] * len(slices)
    moment_reduction = force_reduction = 0.0
    if reinforcements:
        moment_reduction, force_reduction = _reinforcement_terms(slices, xc, yc, R, reinforcements)
    return _gle_fs(slices, fx_list, F_guess=F_guess,
                    moment_reduction=moment_reduction, force_reduction=force_reduction,
                    kh=kh, kv=kv, xc=xc, yc=yc, R=R)


def morgenstern_price_fs(slices, F_guess=1.2, xc=None, yc=None, R=None, reinforcements=None, kh=0.0, kv=0.0):
    """Morgenstern-Price method: f(x) = half-sine, zero at slip-surface ends."""
    fx_list = half_sine_fx(slices)
    moment_reduction = force_reduction = 0.0
    if reinforcements:
        moment_reduction, force_reduction = _reinforcement_terms(slices, xc, yc, R, reinforcements)
    return _gle_fs(slices, fx_list, F_guess=F_guess,
                    moment_reduction=moment_reduction, force_reduction=force_reduction,
                    kh=kh, kv=kv, xc=xc, yc=yc, R=R)


def factor_of_safety(ground_pts, xc, yc, R, layers, method="bishop", n_slices=30,
                      piezo_pts=None, surcharges=None, point_loads=None,
                      weak_boundary=None, reinforcements=None, kh=0.0, kv=0.0):
    """Convenience wrapper: build slices for a trial circle/composite surface and compute FS.
    kh, kv: pseudo-static seismic coefficients (horizontal, vertical). kh > 0 acts
    in the direction of sliding (destabilizing); kv > 0 acts downward (also
    destabilizing; use kv < 0 for an upward/stabilizing vertical seismic component).
    """
    slices = generate_slices(ground_pts, xc, yc, R, n_slices, layers, piezo_pts,
                              surcharges, point_loads, weak_boundary)
    if slices is None:
        return None
    if method == "fellenius":
        moment_reduction = 0.0
        if reinforcements:
            moment_reduction, _ = _reinforcement_terms(slices, xc, yc, R, reinforcements)
        return fellenius_fs(slices, kh=kh, kv=kv, xc=xc, yc=yc, R=R, moment_reduction=moment_reduction)
    elif method == "bishop":
        return bishop_fs(slices, xc=xc, yc=yc, R=R, reinforcements=reinforcements, kh=kh, kv=kv)
    elif method == "spencer":
        return spencer_fs(slices, xc=xc, yc=yc, R=R, reinforcements=reinforcements, kh=kh, kv=kv)
    elif method in ("mp", "morgenstern_price", "morgenstern-price"):
        return morgenstern_price_fs(slices, xc=xc, yc=yc, R=R, reinforcements=reinforcements, kh=kh, kv=kv)
    else:
        raise ValueError(f"Unknown method: {method}")
