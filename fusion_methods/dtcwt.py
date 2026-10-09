# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import functools
import os
import re
import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

from utils.image_utils import fuse_output_dtype

# NumPy 2.0 compatibility shims
if not hasattr(np, "asfarray"):
    def _asfarray_compat(arr, dtype=None):
        target_dtype = dtype or np.float_
        kind = np.dtype(target_dtype).kind
        if kind not in ("f", "c"):
            target_dtype = np.float_
        return np.asarray(arr, dtype=target_dtype)
    np.asfarray = _asfarray_compat

if not hasattr(np, "issubsctype"):
    def _issubsctype_compat(arg1, arg2):
        try:
            dtype1 = np.dtype(arg1) if not isinstance(arg1, np.dtype) else arg1
        except TypeError:
            dtype1 = np.asarray(arg1).dtype
        dtype2 = np.dtype(arg2) if not isinstance(arg2, np.dtype) else arg2
        return np.issubdtype(dtype1, dtype2)
    np.issubsctype = _issubsctype_compat

# Optional imports with checks
try:
    import dtcwt
except ImportError:
    dtcwt = None

# Target working-set size of one column block in the dtcwt filters.
_FILTER_BLOCK_BYTES = 1 << 20


def _column_blocked(fn):
    """Run a dtcwt column filter over cache-sized column blocks.

    dtcwt's colfilter/coldfilt/colifilt filter every column independently
    (the symmetric extension is along rows only) but make one full-image
    pass per filter tap, so a 12 MP frame streams through RAM ~20 times per
    call. Splitting the columns gives the same per-element arithmetic in the
    same order - bit-identical output - with each block staying in cache.
    """
    if getattr(fn, "_openfocus_blocked", False):
        return fn

    @functools.wraps(fn)
    def blocked(X, *filters):
        X = np.asarray(X)
        if X.ndim != 2:
            return fn(X, *filters)
        rows, cols = X.shape
        step = max(16, _FILTER_BLOCK_BYTES // (8 * max(1, rows)))
        if cols <= step:
            return fn(X, *filters)
        return np.concatenate(
            [fn(X[:, c:c + step], *filters) for c in range(0, cols, step)],
            axis=1)

    blocked._openfocus_blocked = True
    return blocked


if dtcwt is not None:
    try:
        import dtcwt.numpy.transform2d as _dtcwt_t2d
        for _name in ("colfilter", "coldfilt", "colifilt"):
            setattr(_dtcwt_t2d, _name,
                    _column_blocked(getattr(_dtcwt_t2d, _name)))
    except (ImportError, AttributeError):  # pragma: no cover - dtcwt layout changed
        pass

# 3x3 window of the max-activity rule and of its majority consistency check.
_WINDOW = np.ones((3, 3), dtype=np.uint8)
# Majority threshold: more than half of the 3x3 window (4.5 of 9 votes).
_MAJORITY = 4


def _activity(mag):
    """3x3 max-filter of coefficient magnitudes, all 6 directions at once.

    Equal to scipy maximum_filter(footprint=3x3x1, mode='reflect'): with a
    3x3 window the reflected neighbour of an edge pixel is the pixel itself,
    which the default dilate border never exceeds. ~6x faster than scipy.
    """
    return cv2.dilate(mag, _WINDOW)


def _fuse_pair(c1, mag1, c2, mag2, act2):
    """Pairwise max-activity fusion with 3x3 majority consistency check.

    Returns the fused coefficients and their magnitudes. |where(W, c1, c2)|
    is where(W, |c1|, |c2|) exactly, so the running magnitude is carried
    along instead of recomputing abs() of the complex array every step.
    """
    initial_mask = np.greater(_activity(mag1), act2)
    # Neighbours voting for c1, zero padded (scipy convolve mode='constant').
    votes = cv2.boxFilter(initial_mask.view(np.uint8), -1, (3, 3),
                          normalize=False, borderType=cv2.BORDER_CONSTANT)
    W = votes > _MAJORITY
    return np.where(W, c1, c2), np.where(W, mag1, mag2)


def _dtcwt_impl(input_source, img_resize, N, use_gpu, thread_count=None):
    """
    DTCWT fusion implementation (Optimized CPU version).
    
    Args:
        input_source: List of images or path to image directory.
        img_resize: Tuple (width, height) or None.
        N: Number of wavelet decomposition levels.
        use_gpu: Ignored in this optimized NumPy version (CPU is used).
        thread_count: Worker threads for the forward transforms
            (None: up to 8, bounded by the CPU count).
    """
    if dtcwt is None:
        raise RuntimeError(
            "DTCWT fusion requires 'dtcwt'. "
            "Please install it via: pip install dtcwt"
        )

    # 1. Load and Preprocess Images
    images = _load_images(input_source, img_resize)
    
    if len(images) < 2:
        raise ValueError("At least two images are required for fusion.")

    # Convert to float32 RGB [0, 1]
    out_dtype = fuse_output_dtype(images)
    images_rgb = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for img in images
    ]
    num_imgs = len(images_rgb)

    if thread_count is None:
        workers = min(8, os.cpu_count() or 1)
    else:
        workers = max(1, int(thread_count))

    transform = dtcwt.Transform2d()

    def forward(job):
        # Everything that depends on one frame alone runs here, in parallel:
        # the transform plus the magnitudes and activity of its coefficients.
        channel_idx, img_idx = job
        t = transform.forward(images_rgb[img_idx][:, :, channel_idx], nlevels=N)
        mags = [np.abs(h) for h in t.highpasses]
        acts = [_activity(m) for m in mags]
        return t.lowpass, t.highpasses, mags, acts

    # 2. Forward transforms run ahead on the pool but are folded strictly in
    # frame order, which the pairwise fusion rule depends on. The bounded
    # look-ahead keeps a few pyramids alive instead of all N x 3 at once.
    jobs = [(c, i) for c in range(3) for i in range(num_imgs)]
    lookahead = 2 * workers
    fused_channels = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque(executor.submit(forward, j) for j in jobs[:lookahead])
        next_job = len(pending)
        for channel_idx, img_idx in jobs:
            lowpass, highs, mags, acts = pending.popleft().result()
            if next_job < len(jobs):
                pending.append(executor.submit(forward, jobs[next_job]))
                next_job += 1

            if img_idx == 0:
                # Low-pass: running sum in frame order, identical to
                # np.mean over the stacked low-passes.
                lowpass_sum = lowpass.copy()
                fused_highs, fused_mags = list(highs), mags
            else:
                lowpass_sum += lowpass
                for level in range(N):
                    fused_highs[level], fused_mags[level] = _fuse_pair(
                        fused_highs[level], fused_mags[level],
                        highs[level], mags[level], acts[level])

            if img_idx == num_imgs - 1:
                # Inverse Transform
                fused_pyramid = dtcwt.Pyramid(lowpass_sum / num_imgs,
                                              tuple(fused_highs))
                fused_channels.append(transform.inverse(fused_pyramid))
                del lowpass_sum, fused_highs, fused_mags

    # 3. Reconstruct Final Image
    fused_img = np.stack(fused_channels, axis=-1)
    
    # Clip and Convert
    fused_img = np.clip(fused_img * 255.0, 0, 255).astype(out_dtype)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    
    return fused_img


def _load_images(input_source: Union[str, List[np.ndarray]], img_resize: Optional[Tuple[int, int]]) -> List[np.ndarray]:
    """Helper to load images from a path or list."""
    images = []
    if isinstance(input_source, str):
        if not os.path.exists(input_source):
             raise ValueError(f"Input path does not exist: {input_source}")
             
        filenames = os.listdir(input_source)
        if not filenames:
            return []
            
        # Infer extension from first file
        suffixes = [os.path.splitext(f)[1] for f in filenames if os.path.splitext(f)[1]]
        img_ext = suffixes[0] if suffixes else ''
        
        glob_pattern = os.path.join(input_source, '*' + img_ext)
        img_paths = glob.glob(glob_pattern)
        
        # Sort numerically if possible
        def sort_key(x):
            nums = re.findall(r"\d+", os.path.basename(x))
            return int(nums[-1]) if nums else x
            
        img_paths.sort(key=sort_key)

        for p in img_paths:
            img = cv2.imread(p)
            if img is not None:
                if img_resize:
                    img = cv2.resize(img, img_resize)
                images.append(img)
    else:
        for img in input_source:
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)
            
    return images