"""Self-contained vendored copy of the Cayley-flow isometry from Learning-ONBs.

Vendored so the kit has no runtime dependency on the sibling repo. The only
modifications to the original sources are import-path fixes (absolute
``infidictionary.*`` imports rewritten to package-relative imports within this
subpackage). See eulerian.EulerianIsometry for the rank-R Cayley isometry.
"""

from .isometry_base import NeuralIsometry, IdentityIsometry
from .eulerian import EulerianIsometry
from .fields import (
    TimeEvolvingField,
    SinusoidalTimeEmbedding,
    NerfSpatioTemporalField,
    LatentBilinearSpatiotemporalField,
)

__all__ = [
    "NeuralIsometry",
    "IdentityIsometry",
    "EulerianIsometry",
    "TimeEvolvingField",
    "SinusoidalTimeEmbedding",
    "NerfSpatioTemporalField",
    "LatentBilinearSpatiotemporalField",
]
