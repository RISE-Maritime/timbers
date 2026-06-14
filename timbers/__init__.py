"""TIMBERS: Time-Modulated Bézier Evolve and Refine Strategy.

GPU/JAX weather routing that co-optimizes route geometry (Bézier control
points) and an explicit time-allocation (speed) profile with separable
CMA-ES, followed by a gradient local-refinement polish. An extension of
the BERS reference method (arXiv 2605.31533).
"""

__version__ = "0.1.0"
