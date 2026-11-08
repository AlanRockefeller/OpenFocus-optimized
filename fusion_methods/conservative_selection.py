"""Experimental source selection; bounded row scratch, no colour-stack copy.

Scores measure local band-pass contrast, not semantic foreground or physical
depth. Conservative selection keeps Hard Pick where confidence, sharpness, or
coherent source structure supports it, and uses one source elsewhere.
"""
import cv2
import numpy as np

from fusion_methods import neural_common as common

ROW_SIZE = 256
DETAIL_SUPPORT_RADIUS = 4
STRUCTURE_RADIUS = 4
STRUCTURE_FRACTION = .70
STRUCTURE_DETAIL_FLOOR = .005
# Gaussian radius 6 + contrast window radius 4 + nearby-detail radius 4 +
# same-source support radius 1.  The extra row is deliberately conservative.
HALO = 16


def scratch_bytes(height, width):
    # Conservative allowance for score/filter temporaries and integer indices.
    return min(height, ROW_SIZE + 2 * HALO) * width * 160


def _score(frame):
    gray = common._plane(frame)
    fine = cv2.GaussianBlur(gray, (5, 5), .7)
    coarse = cv2.GaussianBlur(gray, (13, 13), 2.0)
    energy = cv2.boxFilter((fine - coarse) ** 2, -1, (9, 9))
    level = cv2.boxFilter(np.abs(coarse), -1, (9, 9))
    # Dimensionless contrast; the floor prevents amplification in near-black
    # regions. Pre-smoothing attenuates pixel noise, but is not a noise model.
    return np.sqrt(np.maximum(energy, 0)) / (level + .05)


def selection_rows(probs, frames, fallback='first', confidence=.4,
                   sharpness_advantage=.25, should_cancel=None):
    """Yield (start, stop, selected indices, diagnostic planes).

    Confidence is normalized probability lift above chance plus a relative
    margin.  A decisive model choice or a measured sharpness improvement can
    retain Hard Pick, provided real detail exists locally.  Smooth regions with
    neither kind of evidence use the single fallback source.  Acceptance is
    monotone in both thresholds for fixed frames/probabilities.  All choices
    are original source indices (zero based), never interpolated.
    """
    from core.algorithms import resolve_fusion_parameters
    params = resolve_fusion_parameters('stackmffv3', dict(
        mode='conservative', fallback=fallback, confidence=confidence,
        sharpness_advantage=sharpness_advantage))
    confidence = params['confidence']
    sharpness_advantage = params['sharpness_advantage']
    n, h, w = probs.shape
    base = 0 if fallback == 'first' else n - 1
    for start in range(0, h, ROW_SIZE):
        common.cancel(should_cancel)
        stop = min(h, start + ROW_SIZE)
        lo, hi = max(0, start - HALO), min(h, stop + HALO)
        p = probs[:, lo:hi]
        # Stream the reduction too: argmax over the non-contiguous frame axis
        # can otherwise stage an N-plane contiguous copy inside NumPy.
        top = p[0].copy()
        runner = np.zeros_like(top)
        winner = np.zeros(top.shape, np.int32)
        for i in range(1, n):
            common.cancel(should_cancel)
            better = p[i] > top
            np.maximum(runner, np.minimum(top, p[i]), out=runner)
            np.copyto(winner, i, where=better)
            np.maximum(top, p[i], out=top)
        certainty = ((n * top - 1) / (n * top + 1) * (n + 1) / (n - 1)
                     if n > 1 else top)
        margin = top - runner
        alternative = winner != base
        decisive = ((certainty >= confidence)
                    & (margin >= confidence * .25 * (top + runner)))
        baseline = _score(frames[base][lo:hi])
        best = baseline.copy()
        for i, frame in enumerate(frames):
            common.cancel(should_cancel)
            if i != base and np.any(winner == i):
                score = _score(frame[lo:hi])
                np.copyto(best, score, where=winner == i)
        advantage = (best - baseline) / (baseline + .01)
        observed_detail = best >= .025
        # A sharp edge or texture marks a local subject/detail neighborhood.
        # This lets a decisive Hard choice retain nearby smooth cap interiors,
        # while a uniformly smooth background remains one fallback source.
        detail_supported = cv2.boxFilter(
            observed_detail.astype(np.float32), -1,
            (2 * DETAIL_SUPPORT_RADIUS + 1,) * 2, normalize=False) > 0
        sharper = observed_detail & (advantage >= sharpness_advantage)
        # Dense same-source Hard regions are useful subject evidence even when
        # probability calibration and the fallback-relative score are weak.
        # Focused caps and stems form solid labels; unstable background picks
        # tend to fragment across sources. Detail support remains mandatory so
        # a confidently uniform background cannot become an override region.
        structured = np.zeros_like(alternative)
        structure_kernel = 2 * STRUCTURE_RADIUS + 1
        structure_minimum = STRUCTURE_FRACTION * structure_kernel ** 2
        for i in range(n):
            common.cancel(should_cancel)
            mask = winner == i
            support = cv2.boxFilter(
                mask.astype(np.float32), -1,
                (structure_kernel, structure_kernel), normalize=False)
            structured |= mask & (support >= structure_minimum)
        structure_detail = cv2.boxFilter(
            (best >= STRUCTURE_DETAIL_FLOOR).astype(np.float32), -1,
            (structure_kernel, structure_kernel), normalize=False) > 0
        structure_evidence = structured & structure_detail
        accepted = (alternative & (margin > 0)
                    & (((decisive | sharper) & detail_supported)
                       | structure_evidence))
        # Keep lines and edges, reject isolated choices. Count support for the
        # SAME source; no majority replacement, erosion, or feathering.
        coherent = np.zeros_like(accepted)
        for i in range(n):
            common.cancel(should_cancel)
            mask = accepted & (winner == i)
            support = cv2.boxFilter(mask.astype(np.float32), -1, (3, 3), normalize=False)
            coherent |= mask & (support >= 3)
        selected = np.where(coherent, winner, base).astype(np.int32)
        rows = slice(start - lo, stop - lo)
        yield start, stop, selected[rows], {
            'top1': top[rows], 'margin': margin[rows],
            'sharpness_advantage': advantage[rows], 'accepted': coherent[rows],
            'confidence_accepted': (coherent & decisive)[rows],
            'sharpness_accepted': (coherent & sharper)[rows],
            'structure_accepted': (coherent & structure_evidence)[rows],
            'detail_supported': detail_supported[rows],
            'fallback': ~coherent[rows], 'selected': selected[rows],
        }
    common.cancel(should_cancel)
