"""Direct basis-transformation optimization (no descriptor learning).

See README.md. The idea: instead of jointly learning descriptors *and* a basis
correction through an unsupervised functional-map pipeline, fix the descriptors
(WKS / SHOT / a frozen pretrained net) and learn ONLY an orthonormal change of
basis Q, supervised directly by ground-truth correspondences on a (near-)
isometric dataset (FAUST). The objective aligns the two bases so the ground-truth
functional map becomes the identity, i.e. corresponding points get equal
basis-function values:  Phi_x[corr_x] Q_x  ==  Phi_y[corr_y] Q_y.
"""
