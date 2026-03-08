"""Tests for the diffusion kernel — mass conservation and correctness."""

import numpy as np

from src.diffusion import compute_stable_dt, diffusion_step
from src.fields import allocate, swap_buffers


def _setup_grid(n: int, initial_moisture: np.ndarray | None = None):
    """Allocate fields with interior mask and optional initial moisture."""
    fields = allocate(n)

    # Boundary cells inactive, interior active
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)

    if initial_moisture is not None:
        fields.M.from_numpy(initial_moisture.astype(np.float32))

    return fields


def test_diffusion_conserves_mass():
    """Total moisture in the interior must be conserved under diffusion."""
    n = 64
    rng = np.random.default_rng(42)
    M0 = rng.uniform(0.1, 0.5, (n, n)).astype(np.float32)
    fields = _setup_grid(n, M0)

    mask = fields.mask.to_numpy()
    initial_mass = float(np.sum(M0 * mask))

    D, dx = 0.1, 1.0
    dt = compute_stable_dt(D, dx)

    for _ in range(200):
        diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        swap_buffers(fields.M_new, fields.M)

    final = fields.M.to_numpy()
    final_mass = float(np.sum(final * mask))

    assert abs(final_mass - initial_mass) / initial_mass < 1e-5, (
        f"Mass not conserved: {initial_mass:.8e} -> {final_mass:.8e}"
    )


def test_diffusion_smooths_step():
    """A sharp step function should smooth out over time."""
    n = 32
    M0 = np.zeros((n, n), dtype=np.float32)
    M0[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 1.0
    fields = _setup_grid(n, M0)

    D, dx = 0.5, 1.0
    dt = compute_stable_dt(D, dx)

    for _ in range(500):
        diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        swap_buffers(fields.M_new, fields.M)

    final = fields.M.to_numpy()
    mask = fields.mask.to_numpy()
    interior = final[mask == 1]

    # After diffusion, variance should decrease significantly
    assert np.std(interior) < 0.2, (
        f"Diffusion did not smooth: std={np.std(interior):.4f}"
    )


def test_diffusion_nonnegative():
    """Moisture values must never go negative."""
    n = 32
    rng = np.random.default_rng(99)
    M0 = rng.uniform(0.0, 0.01, (n, n)).astype(np.float32)
    fields = _setup_grid(n, M0)

    D, dx = 0.5, 1.0
    dt = compute_stable_dt(D, dx)

    for _ in range(100):
        diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        swap_buffers(fields.M_new, fields.M)

    final = fields.M.to_numpy()
    assert np.all(final >= 0.0), f"Negative values found: min={final.min()}"


def test_stable_dt_decreases_with_D():
    """Higher diffusivity requires smaller timestep."""
    dx = 1.0
    dt_low = compute_stable_dt(D=0.1, dx=dx)
    dt_high = compute_stable_dt(D=1.0, dx=dx)
    assert dt_high < dt_low


def test_uniform_field_unchanged():
    """A uniform field should remain uniform under diffusion."""
    n = 16
    M0 = np.full((n, n), 0.25, dtype=np.float32)
    fields = _setup_grid(n, M0)

    D, dx = 0.1, 1.0
    dt = compute_stable_dt(D, dx)

    for _ in range(50):
        diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        swap_buffers(fields.M_new, fields.M)

    final = fields.M.to_numpy()
    mask = fields.mask.to_numpy()
    interior = final[mask == 1]
    assert np.allclose(interior, 0.25, atol=1e-6), (
        f"Uniform field changed: max diff={np.max(np.abs(interior - 0.25))}"
    )
