"""Per-fold feature transformation, always fitted on training children only."""

from sklearn.preprocessing import PowerTransformer


def make_preprocessor() -> PowerTransformer:
    """Yeo-Johnson followed by z-scoring (``standardize=True`` does both in one step)."""
    return PowerTransformer(method="yeo-johnson", standardize=True)
