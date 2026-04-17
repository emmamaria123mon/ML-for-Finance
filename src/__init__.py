"""Deep-hedging paper reproduction — shared package.

Reference: S. Fecamp, J. Mikael, X. Warin (2019/2020).
"Deep learning for discrete-time hedging in incomplete markets."
arXiv:1902.05287v4.

Each module maps directly to a section of the paper; see module docstrings.
"""

__all__ = [
    "simulators",
    "payoffs",
    "benchmarks",
    "architectures",
    "losses",
    "training",
    "experiment",
]
