"""Sweep cn × alpha and report regression test metrics.

Run with: uv run python sweep_cn_alpha.py
"""

import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f32)

import numpy as np
import sys
import time
from dataclasses import replace

from src.fields import allocate
from src.flow import compute_flow_fractions, prepare_levels
from src.params import Params
from src.simulate import step_year

N = 200
YEARS = 30  # shorter than 60 to keep sweep feasible
SEED = 42
VEG_THRESHOLD = 5.0  # [g/m²]

# Allocate fields ONCE (Taichi constraint)
FIELDS = allocate(N)


def _generate_rainfall(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = np.zeros(3 * 365, dtype=np.float32)
    for yr in range(3):
        start = yr * 365
        wet_days = rng.choice(365, size=70, replace=False)
        amounts_mm = rng.exponential(4.17, size=70).astype(np.float32)
        for d, amt in zip(wet_days, amounts_mm, strict=True):
            block[start + d] = amt / 1000.0
    return np.tile(block, 20)


def _reset_fields(params):
    """Reset all fields to initial conditions."""
    fields = FIELDS

    mask = np.ones((N, N), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)

    z = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        z[i, :] = (199 - i) * 0.07
    fields.z.from_numpy(z)

    rng = np.random.default_rng(SEED + 1)
    V = np.zeros((N, N), dtype=np.float32)
    interior_coords = np.argwhere(mask == 1)
    chosen = rng.choice(len(interior_coords), size=400, replace=False)
    for idx in chosen:
        i, j = interior_coords[idx]
        V[i, j] = 1.0
    fields.V.from_numpy(V)
    fields.V_new.from_numpy(np.zeros((N, N), dtype=np.float32))

    fields.M.from_numpy(np.full((N, N), 0.1, dtype=np.float32))
    fields.Q_out.from_numpy(np.zeros((N, N), dtype=np.float32))
    fields.Q_daily.from_numpy(np.zeros((N, N), dtype=np.float32))
    fields.Q_annual.from_numpy(np.zeros((N, N), dtype=np.float32))
    fields.I_inf.from_numpy(np.zeros((N, N), dtype=np.float32))
    fields.R.from_numpy(np.zeros((N, N), dtype=np.float32))
    fields.S.from_numpy(np.zeros((N, N), dtype=np.float32))
    fields.S_new.from_numpy(np.zeros((N, N), dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, params.dx, params.p)
    prepare_levels(fields)
    return fields


def _count_bands(V, mask, j=100):
    count = 0
    was_bare = True
    for i in range(N):
        if mask[i, j] == 0:
            continue
        vegetated = V[i, j] > VEG_THRESHOLD
        if vegetated and was_bare:
            count += 1
        was_bare = not vegetated
    return count


def run_one(cn, alpha, years=YEARS):
    params = replace(Params(), cn=cn, alpha=alpha)
    fields = _reset_fields(params)
    rain = _generate_rainfall(SEED)

    V_sum = np.zeros((N, N), dtype=np.float64)
    for yr in range(years):
        step_year(fields, params, rain=rain[yr * 365 : (yr + 1) * 365])
        if yr >= years - 3:
            V_sum += fields.V.to_numpy()

    V_avg = (V_sum / 3).astype(np.float32)
    mask = fields.mask.to_numpy()
    interior = mask == 1

    band_count = _count_bands(V_avg, mask, j=100)
    veg_cells = (V_avg > VEG_THRESHOLD) & interior
    n_veg = int(np.sum(veg_cells))
    n_interior = int(np.sum(interior))
    median_density = float(np.median(V_avg[veg_cells])) if n_veg > 0 else 0.0
    veg_fraction = n_veg / n_interior

    return band_count, median_density, veg_fraction


# Sweep grid — user's derivation says cn=864 ↔ h[mm], cn=8.64e7 ↔ h[m]
# alpha must scale inversely with h to preserve alpha·h product
CN_VALUES = [864, 8640, 86400, 8.64e5, 8.64e6, 8.64e7]
ALPHA_VALUES = [0.8, 8.0, 80.0, 800.0, 8000.0]

# Header
print(f"{'cn':>12s} | {'alpha':>8s} | {'bands':>5s} | {'med_dens':>10s} | {'veg_frac':>10s} | {'pass?':>5s}")
print("-" * 75)
sys.stdout.flush()

for cn in CN_VALUES:
    for alpha in ALPHA_VALUES:
        t0 = time.time()
        bands, med, vfrac = run_one(cn, alpha)
        elapsed = time.time() - t0

        ok_bands = 17 <= bands <= 21
        ok_med = 85.0 <= med <= 115.0
        ok_vfrac = 0.27 <= vfrac <= 0.33
        status = "PASS" if (ok_bands and ok_med and ok_vfrac) else "fail"

        print(
            f"{cn:12.0f} | {alpha:8.1f} | {bands:5d} | {med:10.1f} | {vfrac:10.4f} | {status:>5s}  ({elapsed:.0f}s)"
        )
        sys.stdout.flush()
