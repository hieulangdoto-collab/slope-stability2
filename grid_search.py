"""
Grid & Radius trial-circle search (brute-force baseline) — updated for the
multi-layer / Material-Layer API (v2).
"""
import numpy as np
from le_solver import factor_of_safety, fellenius_fs, generate_slices

FS_MAX = 10.0  # discard trial circles with unrealistically high FS (likely poor geometry)


def grid_search(ground_pts, xc_range, yc_range, r_range, layers,
                 n_slices=25, piezo_pts=None, top_k=5):
    """
    xc_range, yc_range, r_range: (min, max, num_points) tuples.
    layers: list of Layer objects (see le_solver.generate_slices).
    Returns (top_k results, n_tested, n_valid). Each result is (fs_fellenius, xc, yc, R).
    """
    xcs = np.linspace(*xc_range)
    ycs = np.linspace(*yc_range)
    rs = np.linspace(*r_range)

    results = []
    n_tested = 0
    for xc in xcs:
        for yc in ycs:
            for R in rs:
                n_tested += 1
                slices = generate_slices(ground_pts, xc, yc, R, n_slices, layers, piezo_pts)
                if slices is None or len(slices) < 3:
                    continue
                fs = fellenius_fs(slices)
                if fs is None or fs != fs or fs <= 0 or fs > FS_MAX:
                    continue
                results.append((fs, xc, yc, R))

    results.sort(key=lambda t: t[0])
    return results[:top_k], n_tested, len(results)


def refine_with_rigorous_methods(ground_pts, candidates, layers, n_slices=40, piezo_pts=None):
    """Re-evaluate top candidates (found via fast Fellenius screening) with
    Bishop, Spencer, and Morgenstern-Price."""
    refined = []
    for fs_fel, xc, yc, R in candidates:
        fs_b = factor_of_safety(ground_pts, xc, yc, R, layers, method="bishop",
                                 n_slices=n_slices, piezo_pts=piezo_pts)
        fs_s = factor_of_safety(ground_pts, xc, yc, R, layers, method="spencer",
                                 n_slices=n_slices, piezo_pts=piezo_pts)
        fs_mp = factor_of_safety(ground_pts, xc, yc, R, layers, method="mp",
                                  n_slices=n_slices, piezo_pts=piezo_pts)
        refined.append({"xc": xc, "yc": yc, "R": R, "fellenius": fs_fel,
                         "bishop": fs_b, "spencer": fs_s, "mp": fs_mp})
    return refined


if __name__ == "__main__":
    from le_solver import Material, Layer

    H = 10.0
    SLOPE = 2.0
    ground = [(-100.0, H), (0.0, H), (H * SLOPE, 0.0), (120.0, 0.0)]
    mat1 = Material("Clay", c=5.0, phi_deg=20.0, unit_weight=19.0)
    layers = [Layer(mat1, None)]

    print("Scanning grid of trial circles (Fellenius pre-screen)...")
    top, n_tested, n_valid = grid_search(
        ground,
        xc_range=(-10, 30, 21),
        yc_range=(12, 35, 21),
        r_range=(10, 40, 21),
        layers=layers, n_slices=25, top_k=5,
    )
    print(f"Tested {n_tested} trial circles, {n_valid} geometrically valid.\n")

    print("Top 5 candidates (by Fellenius FS), refined with Bishop/Spencer/M-P:")
    refined = refine_with_rigorous_methods(ground, top, layers, n_slices=40)
    for r in refined:
        print(f"  xc={r['xc']:6.2f} yc={r['yc']:6.2f} R={r['R']:6.2f} | "
              f"Fellenius={r['fellenius']:.4f}  Bishop={r['bishop']:.4f}  "
              f"Spencer={r['spencer']:.4f}  M-P={r['mp']:.4f}")
