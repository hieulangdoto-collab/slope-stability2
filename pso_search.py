"""
Particle Swarm Optimization (PSO) for critical slip-surface search — updated
for the multi-layer / Material-Layer API (v2). See grid_search.py for the
brute-force baseline this is meant to replace for large search spaces.
"""
import random
from le_solver import generate_slices, fellenius_fs, factor_of_safety

PENALTY_FS = 50.0  # fitness assigned to geometrically invalid trial circles


def _fitness(pos, ground_pts, layers, n_slices, piezo_pts):
    xc, yc, R = pos
    if R <= 0.5:
        return PENALTY_FS
    slices = generate_slices(ground_pts, xc, yc, R, n_slices, layers, piezo_pts)
    if slices is None or len(slices) < 3:
        return PENALTY_FS
    fs = fellenius_fs(slices)
    if fs is None or fs != fs or fs <= 0:
        return PENALTY_FS
    return fs


def pso_search(ground_pts, bounds, layers,
               n_particles=30, n_generations=60, n_slices=25, piezo_pts=None,
               w=0.6, c1=1.5, c2=1.5, seed=None):
    """
    bounds: dict with 'xc': (min,max), 'yc': (min,max), 'R': (min,max)
    layers: list of Layer objects (see le_solver.generate_slices)
    Returns: (best_position, best_fitness, history)
    """
    rng = random.Random(seed)
    dims = ["xc", "yc", "R"]
    lo = [bounds[d][0] for d in dims]
    hi = [bounds[d][1] for d in dims]

    positions = [[rng.uniform(lo[k], hi[k]) for k in range(3)] for _ in range(n_particles)]
    velocities = [[0.0, 0.0, 0.0] for _ in range(n_particles)]
    pbest = [list(p) for p in positions]
    pbest_fit = [_fitness(p, ground_pts, layers, n_slices, piezo_pts) for p in positions]

    gbest_idx = min(range(n_particles), key=lambda i: pbest_fit[i])
    gbest = list(pbest[gbest_idx])
    gbest_fit = pbest_fit[gbest_idx]

    history = [gbest_fit]

    for gen in range(n_generations):
        for i in range(n_particles):
            for k in range(3):
                r1, r2 = rng.random(), rng.random()
                velocities[i][k] = (w * velocities[i][k]
                                     + c1 * r1 * (pbest[i][k] - positions[i][k])
                                     + c2 * r2 * (gbest[k] - positions[i][k]))
                positions[i][k] += velocities[i][k]
                positions[i][k] = min(max(positions[i][k], lo[k]), hi[k])

            fit = _fitness(positions[i], ground_pts, layers, n_slices, piezo_pts)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = list(positions[i])
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = list(positions[i])
        w *= 0.99
        history.append(gbest_fit)

    return gbest, gbest_fit, history


if __name__ == "__main__":
    from le_solver import Material, Layer

    H = 10.0
    SLOPE = 2.0
    ground = [(-100.0, H), (0.0, H), (H * SLOPE, 0.0), (120.0, 0.0)]
    mat1 = Material("Clay", c=5.0, phi_deg=20.0, unit_weight=19.0)
    layers = [Layer(mat1, None)]

    bounds = {"xc": (-10, 30), "yc": (12, 35), "R": (10, 40)}

    print("Running PSO search (30 particles x 60 generations = 1800 evaluations)...")
    best_pos, best_fs_fellenius, history = pso_search(
        ground, bounds, layers,
        n_particles=30, n_generations=60, n_slices=25, seed=42,
    )
    xc, yc, R = best_pos
    print(f"\nBest trial circle: xc={xc:.2f} yc={yc:.2f} R={R:.2f}")
    print(f"Fellenius FS (fitness used during search) = {best_fs_fellenius:.4f}")

    fs_bishop = factor_of_safety(ground, xc, yc, R, layers, method="bishop", n_slices=50)
    fs_spencer = factor_of_safety(ground, xc, yc, R, layers, method="spencer", n_slices=50)
    fs_mp = factor_of_safety(ground, xc, yc, R, layers, method="mp", n_slices=50)
    print(f"Refined  Bishop FS  = {fs_bishop:.4f}")
    print(f"Refined  Spencer FS = {fs_spencer:.4f}")
    print(f"Refined  M-P FS     = {fs_mp:.4f}")

    print(f"\nConvergence (best FS per generation, every 10th gen):")
    for i in range(0, len(history), 10):
        print(f"  gen {i:3d}: FS = {history[i]:.4f}")
