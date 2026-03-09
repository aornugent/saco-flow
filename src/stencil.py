"""Shared stencil constants for 8-connected grid operations."""

# 8-connected neighbor offsets: NW, N, NE, W, E, SW, S, SE
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Distance multiplier: sqrt(2) for diagonal, 1 for cardinal
DIAG = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]

# Opposite direction index: neighbor k's outflow toward (i,j) is at OPP[k]
OPP = [7, 6, 5, 4, 3, 2, 1, 0]

# 4-connected cardinal neighbors for Laplacian stencils
CARD = [(-1, 0), (1, 0), (0, -1), (0, 1)]
