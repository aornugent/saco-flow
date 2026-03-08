"""Benchmark runner: python -m benchmarks.run"""

import taichi as ti

from benchmarks.diffusion import run as run_diffusion


def main():
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    print("Diffusion stencil benchmark")
    run_diffusion()


if __name__ == "__main__":
    main()
