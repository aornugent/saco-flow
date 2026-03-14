"""Regression test: 60-year banded vegetation pattern formation.

Verifies that the coupled ecohydrological model reproduces banded
vegetation patterns consistent with Baartman et al. (2018) Figure 6:
~19 bands at ~45 m wavelength on a 1.4% slope, vegetated fraction ~1/3,
median band density ~80 g/m².

Domain: 200×200 cells, dx = 5 m (1 km²).
Elevation: z[i,j] = (199 - i) * 0.07 m.
Duration: 60 simulated years.
"""

import numpy as np
import pytest

from src.fields import allocate
from src.flow import compute_flow_fractions
from src.params import Params
from src.simulate import step_year

# -- Domain ------------------------------------------------------------------

N = 200
YEARS = 60
SEED = 42
VEG_THRESHOLD = 5.0  # [g/m²]

# Default Params() encodes Table I / Table II values (dx=5 m, cn=86400).
PARAMS = Params()


# -- Helpers ------------------------------------------------------------------


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

    # Mask: interior = 1, boundary = 0
    mask = np.ones((N, N), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)

    # Elevation: z[i,j] = (199 - i) * 0.07  (slope ~1.4%)
    z = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        z[i, :] = (199 - i) * 0.07
    fields.z.from_numpy(z)

    # Vegetation: 1.0 g/m² in 400 random interior cells, rest 0
    rng = np.random.default_rng(SEED + 1)
    V = np.zeros((N, N), dtype=np.float32)
    interior_coords = np.argwhere(mask == 1)
    chosen = rng.choice(len(interior_coords), size=400, replace=False)
    for idx in chosen:
        i, j = interior_coords[idx]
        V[i, j] = 1.0
    fields.V.from_numpy(V)

    # Soil moisture: 0.1 mm everywhere
    fields.M.from_numpy(np.full((N, N), 0.1, dtype=np.float32))

    # S = 0, Q_out = 0 (zero-initialized by allocate)

    # Precompute flow fractions
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, PARAMS.dx, PARAMS.p)

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


# -- Test ---------------------------------------------------------------------


@pytest.mark.slow
def test_banded_vegetation_60yr():
    """60-year regression: band count, median density, vegetated fraction.

    Acceptance criteria (±5% of paper central values):
      Band count at j=100:       18–20  (central 19)
      Median band density:       76–84  g/m²  (central 80)
      Vegetated fraction:        0.31–0.35  (central 0.33)
    """
    fields = _setup_domain()
    rain = _generate_rainfall(SEED)

    for yr in range(YEARS):
        step_year(fields, PARAMS, rain=rain[yr * 365 : (yr + 1) * 365])

    # -- Metrics at t = 60 yr, interior cells only --
    V = fields.V.to_numpy()
    mask = fields.mask.to_numpy()
    interior = mask == 1

    band_count = _count_bands(V, mask, j=100)

    veg_cells = (V > VEG_THRESHOLD) & interior
    n_veg = int(np.sum(veg_cells))
    n_interior = int(np.sum(interior))
    median_density = float(np.median(V[veg_cells])) if n_veg > 0 else 0.0
    veg_fraction = n_veg / n_interior

    # Diagnostic output (visible on failure)
    diag = (
        f"band_count={band_count}, "
        f"median_density={median_density:.1f} g/m², "
        f"veg_fraction={veg_fraction:.4f}, "
        f"n_veg={n_veg}/{n_interior}"
    )

    assert 18 <= band_count <= 20, f"Band count outside [18, 20]: {diag}"
    assert 76.0 <= median_density <= 84.0, (
        f"Median density outside [76, 84] g/m²: {diag}"
    )
    assert 0.31 <= veg_fraction <= 0.35, (
        f"Vegetated fraction outside [0.31, 0.35]: {diag}"
    )
