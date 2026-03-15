"""Regression test: 60-year banded vegetation pattern formation.

Verifies that the coupled ecohydrological model reproduces banded
vegetation patterns consistent with Baartman et al. (2018) Figure 6:
~19 bands at ~45 m wavelength on a 1.4% slope.

Vegetation density oscillates with the 3-year rainfall cycle, so
metrics are computed on the cycle-averaged V field (last 3 years).

Domain: 200×200 cells, dx = 5 m (1 km²).
Elevation: z[i,j] = (199 - i) * 0.07 m.
Duration: 60 simulated years.
"""

import numpy as np
import pytest

from src.fields import allocate
from src.flow import compute_flow_fractions, prepare_levels
from src.params import Params
from src.simulate import step_year

N = 200
YEARS = 60
SEED = 42
VEG_THRESHOLD = 5.0  # [g/m²]

PARAMS = Params()


def _generate_rainfall(seed: int) -> np.ndarray:
    """Generate 3-year rainfall block, repeated 20× → 60 years.

    Each year: 70 wet days uniformly distributed, rainfall per wet day
    drawn from Exponential(mean=4.17 mm).  70 × 4.17 ≈ 292 mm/yr.
    Returns array of shape (60*365,) in m/day.
    """
    rng = np.random.default_rng(seed)
    block = np.zeros(3 * 365, dtype=np.float32)
    for yr in range(3):
        start = yr * 365
        wet_days = rng.choice(365, size=70, replace=False)
        amounts_mm = rng.exponential(4.17, size=70).astype(np.float32)
        for d, amt in zip(wet_days, amounts_mm, strict=True):
            block[start + d] = amt / 1000.0  # mm → m
    return np.tile(block, 20)  # 3 yr × 20 = 60 yr


def _setup_domain():
    """200×200 grid with slope, boundary mask, and initial conditions."""
    fields = allocate(N)

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

    fields.M.from_numpy(np.full((N, N), 0.1, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, PARAMS.dx, PARAMS.p)
    prepare_levels(fields)

    return fields


def _count_bands(V: np.ndarray, mask: np.ndarray, j: int = 100) -> int:
    """Count bare→vegetated transitions along column j, walking downslope."""
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


@pytest.mark.slow
def test_banded_vegetation_60yr():
    """60-year regression: band count, median density, vegetated fraction.

    Metrics are computed on the cycle-averaged V field (last 3 years)
    to smooth the intra-cycle oscillation driven by the repeated
    3-year rainfall block.

    Acceptance criteria:
      Band count at j=100:       17–21
      Median band density:       85–115  g/m²
      Vegetated fraction:        0.27–0.33
    """
    fields = _setup_domain()
    rain = _generate_rainfall(SEED)

    V_sum = np.zeros((N, N), dtype=np.float64)
    for yr in range(YEARS):
        step_year(fields, PARAMS, rain=rain[yr * 365 : (yr + 1) * 365])
        if yr >= YEARS - 3:
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

    diag = (
        f"band_count={band_count}, "
        f"median_density={median_density:.1f} g/m², "
        f"veg_fraction={veg_fraction:.4f}, "
        f"n_veg={n_veg}/{n_interior}"
    )

    assert 17 <= band_count <= 21, f"Band count outside [17, 21]: {diag}"
    assert 85.0 <= median_density <= 115.0, (
        f"Median density outside [85, 115] g/m²: {diag}"
    )
    assert 0.27 <= veg_fraction <= 0.33, (
        f"Vegetated fraction outside [0.27, 0.33]: {diag}"
    )
