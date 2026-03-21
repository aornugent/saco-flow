"""Poisson-cluster rainfall generator for semi-arid Mediterranean climates.

Reproduces the statistical structure of the SE Spain record used in
Baartman et al. (2018) Figure 5: ~292 mm/yr annual total, events up to
~65 mm, multi-day storm clustering, and a pronounced wet season.

The generator is parameterised by summary statistics rather than distribution
constants, so it can be calibrated against a reference rainfall sequence.
"""

import numpy as np


def generate_rainfall(
    seed: int,
    *,
    block_years: int = 3,
    n_tiles: int = 20,
    target_annual_mm: float = 292.0,  # [mm/yr]
    n_storms: float = 8.0,  # mean storm systems per wet season
    wet_season_days: int = 180,  # wet season length [days]; 365 = year-round
    storm_duration_p: float = 0.25,  # geometric(p) → mean 1/p days
    frac_intense: float = 0.3,  # fraction of intense cells within a storm
    intense_shape: float = 0.5,  # gamma shape for intense cells
    intense_scale: float = 30.0,  # gamma scale for intense cells  [mm]
    moderate_shape: float = 1.5,  # gamma shape for moderate cells
    moderate_scale: float = 6.0,  # gamma scale for moderate cells [mm]
    n_dry_events: float = 15.0,  # mean scattered events in dry season
    dry_scale: float = 3.0,  # exponential scale for dry-season events [mm]
) -> np.ndarray:
    """Generate a block_years rainfall block, tiled n_tiles times.

    Poisson-cluster process:
      - Wet season: clustered multi-day storm systems
      - Dry season (if wet_season_days < 365): scattered light events
      - Normalised to target_annual_mm per year

    When wet_season_days=180 (default), the wet season is Oct–Mar and
    dry season Apr–Sep, matching Mediterranean climate.  Set to 365
    for year-round storms (useful for short feedback tests).

    Returns array of shape (block_years * n_tiles * 365,) in m/day.
    """
    rng = np.random.default_rng(seed)
    block_days = block_years * 365
    block = np.zeros(block_days, dtype=np.float64)

    for yr in range(block_years):
        offset = yr * 365

        # Wet season: centred on Oct–Mar when seasonal, full year otherwise
        if wet_season_days >= 365:
            wet_days_pool = list(range(365))
        else:
            # Oct–Mar → days 270–365 + 0–(wet_season_days - 95 - 1)
            half = wet_season_days // 2
            wet_days_pool = list(range(0, half)) + list(range(365 - half, 365))

        n_sys = rng.poisson(n_storms)
        storm_starts = rng.choice(
            wet_days_pool, size=min(n_sys, len(wet_days_pool)), replace=False
        )

        for start in storm_starts:
            duration = rng.geometric(storm_duration_p)
            for d in range(duration):
                day = offset + (int(start) + d) % 365
                if 0 <= day < block_days:
                    if rng.random() < frac_intense:
                        block[day] += rng.gamma(intense_shape, intense_scale)
                    else:
                        block[day] += rng.gamma(moderate_shape, moderate_scale)

        # Dry season: scattered light events (skip if year-round wet)
        if wet_season_days < 365:
            wet_set = set(wet_days_pool)
            dry_pool = [d for d in range(365) if d not in wet_set]
            n_dry = rng.poisson(n_dry_events)
            dry_chosen = rng.choice(
                dry_pool, size=min(n_dry, len(dry_pool)), replace=False
            )
            for d in dry_chosen:
                day = offset + d
                if 0 <= day < block_days:
                    block[day] += rng.exponential(dry_scale)

    # Normalise to target annual total
    total_mm = np.sum(block)
    if total_mm > 0:
        block *= (target_annual_mm * block_years) / total_mm

    # mm → m, tile to full duration
    return np.tile((block / 1000.0).astype(np.float32), n_tiles)
