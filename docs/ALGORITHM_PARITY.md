# Algorithm parity: OpenFocus vs Helicon Focus

Reference: the published Helicon Focus 9 user guide
(`heliconsoft.com/focus/help9/english/HeliconFocus.html`) and the parameter
overview (`heliconsoft.com/helicon-focus-main-parameters`). Only publicly
documented behaviour is compared; no binaries, code or analysis material from
that product is used or included.

"Parity" here means **behavioural capability**, not a bit-exact reproduction:
our methods are independent implementations, and the synthetic benchmark in
`tools/synthetic_benchmark.py` is the arbiter of output quality.

## Fusion methods

| Capability | Helicon | OpenFocus | State |
|---|---|---|---|
| Weighted-average method | Method A | `openfocus_a` | aligned |
| Depth-map pick method | Method B | `openfocus_b` | aligned |
| Pyramid method | Method C | `openfocus_c` | aligned |
| Radius (A, B) | yes | `--kernel` / slider | aligned |
| Smoothing (A, C: blend) | yes | `--blend-smoothing` / spin box | aligned (v1.59) |
| Smoothing (B: depth map) | yes | `--depth-smoothing` / spin box | aligned (v1.54) |
| No Radius for C | yes | radius slider hidden for C | aligned (v1.59) |
| Brightness compensation across a stack | yes, automatic | `exposure_match` (default on) | aligned (v1.57) |
| Sorting order asc/desc/auto | yes | `--sort asc\|desc\|off` | aligned |
| 8/16-bit pipeline | yes | normalise/quantise end to end | aligned |

## Alignment

| Capability | Helicon | OpenFocus | State |
|---|---|---|---|
| Fail-fast automatic alignment | yes | `openfocus` aligner | aligned |
| Vertical/horizontal shift tolerance | separate params (`-va`, `-ha`) | one `--align-shift-limit` | **partially aligned** — one knob covers both axes |
| Rotation tolerance | separate param (`-ra`) | `_OF_MAX_ROTATION_DEG` (8°) | aligned (v1.59) |
| Magnification tolerance | separate param (`-ma`) | `_OF_MAX_SCALE_DELTA` (±5%) | aligned (v1.59) |
| Brightness tolerance | separate param (`-ba`) | gain clamped to 0.5–2.0 | approximate |
| Interpolation choice | selectable (`-im`) | fixed Lanczos4 warp | **not aligned** |
| Auto-crop of aligned edges | yes, toggleable | always on (valid-region intersection) | aligned (no toggle) |
| Depth-map feathering | separate param (`-dmf`) | `--depth-smoothing` (B) | aligned for B only |

## Output and workflow (not algorithm, listed for completeness)

| Capability | Helicon | OpenFocus | State |
|---|---|---|---|
| Retouch brush (copy from source) | yes | retouch dialog | aligned (v1.52) |
| Clone / erase brushes | yes | not implemented | **not aligned** |
| Depth map export | histogram + map | depth map dialog + colormap | aligned |
| Scale bar / text | yes | scale bar + labels | aligned |
| 3D model export | yes | textured OBJ | aligned (v1.55) |
| Batch processing | yes | batch manager | aligned |
| Split stacks (by count/time/exposure/focus) | yes | not implemented | **not aligned** |
| Folder monitor | yes | not implemented | **not aligned** |
| Raw-in-DNG-out | yes (Pro) | RAW input only, no DNG output | **not aligned** |
| Animated stack export | yes (Pro) | GIF export | partially aligned |
| Export aligned layers | yes (Pro) | not implemented | **not aligned** |

## Open items

1. **Independent shift tolerances.** Helicon lets horizontal and vertical
   tolerances differ (e.g. a panorama rail drifting only horizontally). We
   accept a single fraction for both. Adding two parameters is a small
   change, but nothing in the measurement so far shows a case where the
   single limit produces a worse result, so it is deliberately not done yet.
2. **Interpolation choice.** We warp with Lanczos4. Helicon exposes the
   interpolator; for heavily scaled stacks a cheaper/sharp model may matter.
3. **Clone / erase brushes.** Our retouch copies from a source frame only.
4. **Split stacks / folder monitor / aligned-layer export.** Workflow
   features, not algorithms.
5. **DNG output.** Requires a raw development pipeline; we read RAW but
   write 8/16-bit raster.

## How the measurements were made

`tools/synthetic_benchmark.py` renders a ground-truth scene through a known
depth map, so PSNR/SSIM/ghost figures are against the true image rather than
a reference product's output. Method A's lower PSNR on that benchmark is
expected and inherent: a weighted average blends neighbouring frames, so it
cannot reach the oracle pick. Forcing its weights to a hard pick turns it
into Method B (measured: 23.5 dB → 39.1 dB when the weight softmax is
sharpened), which is a different method, not a fix.
