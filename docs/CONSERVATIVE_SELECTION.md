# Experimental StackMFF V3 Conservative Selection

This opt-in fusion policy retains Hard Pick in locally detailed subject regions
when either model confidence or measured sharpness supports the choice, and
uses one processed source frame in background-like regions. Soft Blend remains
the default. The ONNX model,
weights, Soft/Hard math, and independent Auto Background feature are unchanged.
No automatic background replacement is invoked by this policy.

## Controls

The GUI exposes these controls only when Conservative Selection is selected.
The normal parameter registry handles CLI, batch JSON, settings, and projects.

| Parameter | Default / range | Meaning |
|---|---|---|
| `mode` | `soft`, `hard`, `conservative` | V3 only |
| `fallback` | `first`; `first` or `last` | One source in the actual processed order. GUI completion report and CLI output show its 1-based source number. |
| `confidence` | 0.40; 0–1 | Higher values require more decisive model probabilities for the confidence evidence path. |
| `sharpness_advantage` | 0.25; 0–10 | Higher values require greater normalized local contrast improvement for the independent sharpness evidence path. |

First/last do **not** necessarily mean frontmost/rearmost focus. Folder ordering,
EXIF focus-step ordering, explicit file order, and any preprocessing determine
the processed order. Check the source list. The default is first for a predictable
reference; choose last when its background looks more natural. No automatic
fallback choice or semantic foreground segmentation is attempted.

```sh
python main.py --input /path/to/stack --output /tmp/conservative.tif \
  --method stackmffv3 --ai-mode conservative \
  --param 'fallback="last"' --param confidence=0.4 --param sharpness_advantage=0.25
```

Use the project's normal CLI entry point if installed. Do not pass
`--auto-background` for comparisons. Apply identical alignment/exposure settings
to all modes. The generic `--param` syntax is the established CLI parameter
interface; no additional flags are required. A batch configuration's existing
`fusion_parameters` object can contain:

```json
{"mode":"conservative", "fallback":"first", "confidence":0.4,
 "sharpness_advantage":0.25}
```

Nonfinite values, booleans used as numbers, out-of-range values, unknown keys,
and unsupported fallback choices are rejected. Old recipes acquire defaults.

## Policy and boundaries

V3's existing two-scale inference runs without Soft Blend's probability
smoothing. For N > 1 frames, confidence requires
`((N*p1 - 1)/(N*p1 + 1)) * ((N + 1)/(N - 1)) >= confidence`
and `(p1 - p2)/(p1 + p2) >= confidence/4`. The first score measures probability
relative to the uniform baseline, normalized to 1 at a decisive winner. The
relative runner-up margin rejects ambiguous ties without a fixed absolute
margin that becomes prohibitive in long stacks. These scores do **not** calibrate
focus probability or remove all dependence on stack length. Softmax can be decisive when every image is blurred.

A separate score uses luma normalized to 0–1, the RMS difference of Gaussian
filters (sigma 0.7 and 2.0) over a 9×9 neighborhood, divided by local brightness
plus 0.05. An observed-detail pixel has score at least 0.025. A 9×9 support
window extends that evidence slightly into nearby smooth pixels. Within this
supported region, an alternative is retained when the confidence tests above
pass, when `(alternative - fallback)/(fallback + 0.01) >=
sharpness_advantage`, or when at least 70% of a 9×9 neighborhood has the same
Hard source label and its relaxed local-detail score reaches 0.005. The last
path preserves coherent focused subjects whose
probability and contrast scores are individually weak, while fragmented source
choices in unstable background do not qualify.
These experimental dimensionless thresholds are bit-depth independent and
partly illumination normalized. They are **not** invariant to magnification,
resampling, sensor noise, clipping, registration errors, or illumination changes.
Pre-smoothing reduces single-pixel noise; structured noise may still pass.

At least three accepted pixels from the same source in a 3×3 neighborhood must
support a choice. This retains supported one-pixel lines without replacing labels
by a majority or opening/eroding masks. The score and support neighborhoods let
nearby sharp edges support smooth pixels locally. A broad smooth cap interior can
remain fallback; there is no semantic mask to propagate across its entire surface.
Acceptance is monotone in either threshold for fixed inputs and probabilities.

Decisive output copies processed source values, preserving 16-bit low bits.
There is no feathering or invented color. Exposure matching and alignment happen
in the existing pipeline, before this policy; provenance refers to those
processed frames. Existing common-valid-region cropping still applies. Direct
API callers must provide aligned, valid image regions; this policy cannot infer
validity from black pixel values. Residual registration/exposure differences can
produce selection boundaries and should be corrected upstream.

Scoring streams 256-row stripes with a 16-row halo, one source at a time; it does
not allocate another N-frame color stack. Additional admitted scratch is at most
`min(height,288) * width * 160` bytes. Tile inference still uses the existing
resource planner and cancellation hooks. Conservative overlap is at least 32
pixels (capped by half a tile). Conflicting tile picks use the highest interior
weight, with earlier tiles winning ties. Pixels are never averaged across
conflicting tiles. Local scoring matches whole-frame scoring under fixed
probabilities away from padding; the attention network itself sees different
context per tile, so real tiled/whole results need not be identical and visible
selection seams remain possible. Soft/Hard overlap behavior is unchanged.

Conservative focus-depth export is explicitly rejected, including in the depth
dialog. Select Soft/Hard there for a separate focus-position estimate. Debug
`selected.npy` is a zero-based **source provenance** map, not geometric depth or
semantic foreground/background. Soft/Hard depth semantics are unchanged.

A fallback can contain out-of-focus foreground spill or occlusion. If no source
shows clean background at a pixel, no selector can recover it. This mode does not
solve that physical limitation and does not promise to remove every halo.

## Reproducible comparison

```sh
python tools/compare_conservative.py --input assets/demo_stack --align \
  --output /tmp/v3-demo --diagnostics
python tools/compare_conservative.py --input /path/to/aligned/mushroom-stack \
  --max-size 0 --output /tmp/v3-mushroom --diagnostics \
  --crop cap-margin,100,200,128,128 --crop gills,300,400,128,128 \
  --crop hairs,400,600,128,128 --crop overlap,200,700,128,128 \
  --crop bokeh,800,200,128,128
```

Crop coordinates must be chosen for the user's image. Without crop arguments,
five generic regions are used; these are not automatically identified anatomy.
The first command resizes to a 384-pixel longest edge before one common alignment.
Use `--max-size 0` for original resolution. All eight modes/settings receive the
same prepared pixels, saved under `prepared/`; Auto Background, sharpening,
base correction, and exposure matching are off. The sheet contains full-image
thumbnails and identical 100% crops of the **prepared** stack. Outputs are TIFF;
only the sheet is an 8-bit preview.

`--diagnostics` writes disk-backed top-1, margin, sharpness-advantage,
confidence-supported, sharpness-supported, structure-supported, nearby-detail, final acceptance,
fallback-use, and selected-source `.npy` maps, including correct tile ownership.
It is opt-in and can use substantial disk space. `--tile-size 256` compares tiled
fusion. `report.json` records settings, runtime, fallback/override percentages,
override connected components, adjacent source-label disagreement, and mean
absolute difference from Hard Pick in a common edge band. Components and label
disagreement refer to the whole image, **not** a semantic background mask. The
edge difference is a change measure, not an artifact score. Process peak RSS is
a cumulative high-water reading including input, ONNX, previous runs and debug
maps; it is not an isolated per-mode memory comparison.

Review cap margins, gills, hairs, overlaps and background at full resolution on
representative real mushroom stacks before adopting this experimental policy.

See [implementation results and measured limitations](CONSERVATIVE_SELECTION_RESULTS.md) for this checkout’s completed comparisons.
