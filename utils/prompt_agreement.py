"""Prompt-agreement confidence for binary segmentation masks."""

import itertools

import numpy as np


def _binary_mask(mask):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(
            "each primary mask must be two-dimensional, got {}".format(
                array.shape
            )
        )
    if array.dtype != np.bool_:
        unique = np.unique(array)
        if not np.isin(unique, (0, 1)).all():
            raise ValueError("primary masks must be binary")
        array = array.astype(bool)
    return array


def pairwise_mask_iou_matrix(masks, empty_union_iou=1.0):
    """Return a symmetric, full-image IoU matrix for binary masks."""
    masks = tuple(_binary_mask(mask) for mask in masks)
    if len(masks) < 2:
        raise ValueError("prompt agreement requires at least two masks")
    if not 0.0 <= empty_union_iou <= 1.0:
        raise ValueError("empty_union_iou must be within [0, 1]")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks[1:]):
        raise ValueError("all primary masks must have the same shape")

    matrix = np.eye(len(masks), dtype=np.float64)
    for left, right in itertools.combinations(range(len(masks)), 2):
        intersection = int(np.logical_and(masks[left], masks[right]).sum())
        union = int(np.logical_or(masks[left], masks[right]).sum())
        value = empty_union_iou if union == 0 else intersection / union
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


def summarize_prompt_agreement(
    masks,
    confidence_threshold=None,
    empty_union_iou=1.0,
):
    """Return the minimum pairwise IoU and its strict threshold flag."""
    if confidence_threshold is not None and not (
        0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must be within [0, 1]")

    matrix = pairwise_mask_iou_matrix(
        masks, empty_union_iou=empty_union_iou
    )
    count = matrix.shape[0]
    minimum = float(matrix[np.triu_indices(count, k=1)].min())
    return {
        "pairwise_mask_iou": matrix,
        "minimum_pairwise_iou": minimum,
        "confidence_threshold": confidence_threshold,
        "unconfident": bool(
            confidence_threshold is not None
            and minimum < confidence_threshold
        ),
    }
