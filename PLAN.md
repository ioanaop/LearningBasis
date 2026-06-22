# Learned Orthonormal Basis Correction for Functional Maps

## Goal
Insert a learned, orthonormal change-of-basis Q_θ between the raw LBO
eigenbasis and the functional-map solver in DeepShapeMatchingKit, so the
basis adapts to improve matching while staying exactly orthonormal w.r.t.
the mesh mass inner product. Q_θ is the Cayley-integrated rank-R flow from
the Learning-ONBs repo (Solomon et al., "Learning Orthonormal Bases for
Function Spaces").

## Source of Q_θ
`../Learning-ONBs/infidictionary/neural_isometries/eulerian.py`,
class `EulerianIsometry`. Its Cayley step operates on FUNCTION VALUES at N
sample points with a per-point measure weight `logabsdet`. Key insight:
that measure weight IS the mesh mass. Set exp(logabsdet) = mass per vertex,
coords = vertices, values = eigenfunctions. The Cayley step is then
orthonormal w.r.t. the mass-weighted inner product BY CONSTRUCTION.

## Injection point in the kit
The fmap solver touches the basis only through:
- `evecs`        = Φ
- `evecs_trans`  = Φ^T M   (mass-weighted projector; see utils/geometry_util.py)
- `Mk`           = Φ^T M Φ
After computing corrected Φ̃ = Q_θ Φ, recompute these three and pass them on.
Nothing else changes.

## Loss
FAUST ULRSSM config uses SURFMNetLoss (standard) with w_lap=0.0, and the
plain FasterRegularizedFMNet solver (NO mass matrix threaded).

FIRST BUILD = COEFFICIENT SPACE. Apply Q as a k×k rotation on the spectral
coefficients. Because raw LBO eigenfunctions are already M-orthonormal,
Mk = Φ^T M Φ = I, so the mass-weighted and standard inner products COINCIDE
in coefficient space. Therefore:
  - evecs_trans_corrected = Q @ evecs_trans   (Q is k×k)
  - the plain FasterRegularizedFMNet stays valid UNCHANGED
  - SURFMNetLoss (standard orthogonality) stays valid UNCHANGED
  - keep w_lap=0.0 (commutativity term is ill-defined off the LBO eigenbasis)
Do NOT switch to HS_SURFMNetLoss or ExpandedResolventFMNet for the first build.
Those are only needed for the later VERTEX-SPACE version where Mk != I.

## Architecture decisions (already made — do not revisit)
- SYMMETRIC: one shared Q_θ network applied to BOTH shapes.
- Condition the generator's spatial MLP on INTRINSIC features (DiffusionNet
  features), NOT raw xyz coords, so the correction is consistent across the pair.
- Differentiate through the unrolled Cayley steps via gradient checkpointing
  (already supported in EulerianIsometry). No continuous adjoint.
- Start small: rank R=10–20, L=20 steps, T=1. FAUST.

## First checkpoint that matters
A `Q = identity` ablation switch. With Q=I the whole pipeline must reproduce
vanilla ULRSSM numerically. This is the correctness test before any training.

## Coefficient-space adaptation of EulerianIsometry
The flow acts on function VALUES at N points with measure weight logabsdet.
For coefficient space: treat the k coefficient slots as the "points" (N=k),
set the measure weight uniform (logabsdet=0, since Mk=I means uniform weight),
feed the k basis functions' coefficient representation. Output is the rotated
coefficient basis = a k×k orthogonal Q. This is simpler than vertex space and
is the FIRST target. Vertex space (coords=verts, logabsdet=log(mass)) is the
SECOND, more powerful version — defer it.