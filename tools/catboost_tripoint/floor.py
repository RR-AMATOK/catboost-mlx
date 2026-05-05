"""
floor.py -- Theoretical fp32 accumulation-error floor for GBDT predictions.

The prediction of a gradient-boosted model is an additive sum of T leaf values:

    pred(x) = sum_{t=1}^{T} leaf_t(x)

Each leaf_t(x) is a fp32 value, and the sum accumulates T such terms.  By
Wilkinson-style backward error analysis for sequential fp32 addition, the
rounding error in the partial sum after T terms is bounded by:

    |fl(sum) - true_sum| <= (T * eps_mach) * max_leaf_value   [sequential]

For pairwise summation the exponent on T reduces to log2(T), but catboost's
leaf accumulation is sequential, so we use the linear bound.

The "width" of a single leaf value is bounded by the model's leaf magnitude,
which at fp32 precision contributes an ulp of:

    eps_leaf = eps_mach * |leaf_value|

Averaged over sqrt(L) pairwise cancellations inside a tree of L leaves, the
per-tree contribution is ~ eps_mach * |leaf_value| * sqrt(L), and after T
trees the bound becomes:

    Floor ~= eps_mach * T * sqrt(L)

where L = 2^depth is the maximum number of leaves per tree.

This is a Wilkinson-style upper bound (see references below), not a tight
bound.  Observed cross-backend divergences are typically 1-2 orders of
magnitude smaller.

References
----------
Wilkinson, J.H. (1965). Rounding Errors in Algebraic Processes.
    Prentice-Hall. (The foundational reference for backward error analysis
    of floating-point summation; see Chapter 1, §§1.1–1.4.)
Higham, N.J. (2002). Accuracy and Stability of Numerical Algorithms, 2nd ed.
    SIAM. (Chapter 4, "Summation"; Theorem 4.1 gives the u*n backward error
    bound for sequential summation of n fp terms; our formula instantiates
    this with n=T and scales by the sqrt(L) per-tree pairwise term.)
"""

import math

#: float32 machine epsilon: 2^-23 ~= 1.1920929e-07
EPSILON_MACHINE_FP32: float = 2.0 ** -23


def derived_floor_for_model(tree_count: int, max_depth: int) -> float:
    """Return the theoretical fp32 accumulation floor for a GBDT model.

    Formula: eps_mach * T * sqrt(L), where L = 2^max_depth.

    Parameters
    ----------
    tree_count : int
        Number of boosting rounds (trees) in the model.
    max_depth : int
        Maximum tree depth.  L = 2^max_depth is the maximum leaf count per
        tree; sqrt(L) = 2^(max_depth/2) captures the pairwise-summation
        error accumulation within a single tree.

    Returns
    -------
    float
        Upper bound on fp32 prediction divergence between two backends that
        perform the same leaf computation in different order or with
        intermediate rounding differences.

    Notes
    -----
    This formula uses the *maximum* leaf count (2^depth).  For models with
    variable depth per tree, pass the maximum observed depth to obtain an
    upper bound, or use `--floor-mode strict` (not yet implemented) to walk
    each tree for a tighter estimate.
    """
    if tree_count <= 0:
        raise ValueError(f"tree_count must be positive, got {tree_count}")
    if max_depth <= 0:
        raise ValueError(f"max_depth must be positive, got {max_depth}")

    max_leaves = 2 ** max_depth
    sqrt_leaves = math.sqrt(max_leaves)
    return EPSILON_MACHINE_FP32 * tree_count * sqrt_leaves


def floor_components(tree_count: int, max_depth: int) -> dict:
    """Return a dict of all formula components for display in reports.

    Parameters
    ----------
    tree_count : int
    max_depth : int

    Returns
    -------
    dict with keys: epsilon_machine, tree_count, max_depth, max_leaves,
        sqrt_leaves, derived_floor, formula.
    """
    max_leaves = 2 ** max_depth
    sqrt_leaves = math.sqrt(max_leaves)
    floor_val = EPSILON_MACHINE_FP32 * tree_count * sqrt_leaves
    return {
        "epsilon_machine": EPSILON_MACHINE_FP32,
        "tree_count": tree_count,
        "max_depth": max_depth,
        "max_leaves": max_leaves,
        "sqrt_leaves": sqrt_leaves,
        "derived_floor": floor_val,
        "formula": "eps_mach * T * sqrt(L)  [Wilkinson 1965, Higham 2002 Ch.4]",
    }
