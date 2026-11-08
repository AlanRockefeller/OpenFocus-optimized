# Conservative Selection implementation results — 2026-11-06

> **November 7 subject-preserving revision:** real mushroom review showed that
> the original confidence-AND-sharpness gate retained blurred fallback pixels
> across the subject. The policy now keeps Hard Pick where decisive confidence,
> observed sharpness, or a dense same-source 9×9 region supports local detail.
> On the reviewed 15-frame native mushroom stack, difference from Hard fell
> from 24.06% to 2.00% of pixels and mean native difference fell from 396.4 to
> 33.4, while 46.10% of the image remained the last-frame fallback. On the
> bundled jewelry comparison, the two smooth-background crops remained 97.98%
> and 98.13% fallback. Historical measurements below describe the original
> policy unless noted.

Implemented on `feature/v3-conservative-selection`, without a commit, push, PR,
model-weight change, or training-repository change. This is an experimental
selection policy, not a demonstrated general replacement for Soft Blend or Hard
Pick. Real mushroom validation is pending: the bundled stacks are jewelry and
a keyboard, not representative cap/gill/hair stacks.

## Implementation and defaults

- V3-only `mode=conservative`; existing default remains `soft`.
- `fallback=first` by default; `last` selects the last **processed** source.
  The completion report/console identifies its 1-based number.
- `confidence=0.40`, range 0–1: normalized probability lift over the uniform
  baseline, together with a relative top-versus-runner-up margin.
- `sharpness_advantage=0.25`, range 0–10: improvement in normalized local
  band-pass contrast; an independent absolute contrast floor also applies.
- Selection uses source pixels, bounded row scoring, small same-source support,
  and deterministic tile ownership. Conservative overlap is at least 24 pixels.
- GUI conditional controls, 12 language packs, parameter persistence, CLI
  `--ai-mode`/`--param`, and batch JSON use the existing registry. No new config
  format. Depth export explicitly rejects this mode; diagnostic indices are
  provenance, not focus depth. Auto Background is unchanged and is not invoked.

Code is concentrated in `fusion_methods/conservative_selection.py`, the V3
backend/plumbing, parameter registry/UI, CLI validation, depth validation, and
completion reporting. `tests/test_conservative_selection.py` adds deterministic
policy, GUI, CLI/batch, persistence, resource, cancellation, precision, and tile
coverage. `tools/compare_conservative.py` produces reproducible diagnostics.
The English manual and fusion-method guide link the detailed policy document.

The initial 0.60/0.50 settings left visibly blurred jewelry detail in the fallback.
0.40/0.25 recovered more detail. The original absolute probability formulation
also rejected too many alternatives in the 32-frame stack; normalized probability
lift plus a relative margin reduced this stack-length problem. These adjustments
are based on the available examples, not a calibrated probability model.

## Verification

Environment: isolated `/tmp/openfocus-conservative-venv`, Python 3.13.15,
NumPy 2.5.4, OpenCV 5.0.0.93, ONNX Runtime 1.23.2, PyQt6 6.12.0, CPU inference.
Use the following headless Qt environment on this machine:

```sh
export QT_QPA_PLATFORM=offscreen
export QT_QPA_PLATFORMTHEME=generic
export QT_STYLE_OVERRIDE=Fusion
PY=/tmp/openfocus-conservative-venv/bin/python
$PY -m pytest tests/test_conservative_selection.py tests/test_stackmffv3.py \
  tests/test_neural_streaming.py tests/test_algorithms.py tests/test_cli.py \
  tests/test_algorithm_surfaces.py tests/test_parameter_memory.py \
  tests/test_i18n_packs.py tests/test_i18n_placeholders.py tests/test_gui_units.py
$PY -m pytest
```

The final targeted run: **222 passed, 1 skipped**, 19.64 seconds. The suite tests
blurry/flat/textured backgrounds, dark/light synthetic spill, observed-detail
selection, smooth interiors near edges, one-pixel lines, isolated predictions,
threshold monotonicity, first/last fallback, 8/16-bit and grayscale/non-contiguous
inputs, odd sizes, row halos, cancellation, resource admission, real CPU ONNX,
whole/tiled provenance, settings, conditional GUI controls, batch configuration
including actual 16-bit CLI output, malformed parameters and depth rejection.

The first unrestricted full run: 2,975 passed, 13 skipped, three translation
coverage failures, all subsequently fixed and covered by the targeted run.
The initial restricted run was interrupted because local-socket GUI tests could
not run correctly in the sandbox; full runs use local-socket access. The first
headless attempt also needed `QT_QPA_PLATFORMTHEME=generic` to avoid the desktop
GTK theme trying to open a display. GPU inference and interactive visual GUI
operation were not tested.

Final full run after the translation fixes: **2,979 passed, 13 skipped**, five
warnings, 163.48 seconds. Warnings came from Pillow TIFF writer cleanup (three)
and TIFF resolution metadata (two). A final bounded top-two probability
reduction was then verified by the focused policy tests, including reference
checks with 2, 32 and 200 sources; this removes possible implicit N-plane
reduction staging without changing the selection scores. The final focused check passed **41 tests with 12 expected skips**, including
all 35 Conservative Selection tests. The 13 full-suite skips comprise six
PyTorch-dependent tests (PyTorch absent), two unsupported XPU-provider cases,
three Windows path cases, one macOS `ditto` case and one PowerShell recovery case.
CPU ONNX tests ran; GPU execution did not.

On continuation (November 7), additional changes to frame storage, source-frame
materialization, workers and frame-storage tests were already present in the
shared working tree. They were preserved and are not part of this feature's
implementation. The full-suite result above predates those changes. A check of
the current tree covering Conservative Selection, V3 and neural streaming passed
**78 tests in 10.06 seconds** with a dedicated temporary root. An initial concurrent
check hit missing `/tmp/openfocus-run-2` directories under the new PID-based
storage cleanup; isolating `TMPDIR` removed that sandbox-process collision. Those
storage changes were not modified by this task.

A separate regression comparison loaded the original `HEAD` versions of both
V3 and the fusion dispatcher and compared them against this tree on the exact
six-frame prepared jewelry stack. **Soft and Hard were byte-identical in both
whole-frame and tiled runs**, maximum difference 0 in all four cases. The script
and JSON evidence are in `.probe/conservative-selection/`.

## Measurements and visual review

Local artifacts: `.probe/conservative-selection/{demo,demo-tiled,native,keyboard}`.
Each has prepared TIFF inputs, eight output TIFFs, a comparison sheet, diagnostic
`.npy` maps, and `report.json`. Auto Background, exposure matching, base correction,
and sharpening were off. Jewelry/keyboard were aligned once with OpenFocus;
the exact prepared pixels were reused across modes/settings. Smaller runs were
resized **before** alignment. Native jewelry was not resized.

| Prepared stack | Shape / tiles | Soft seconds | Hard seconds | Conservative first / last seconds | First / last fallback % |
|---|---|---:|---:|---:|---:|
| Jewelry, 6 frames | 380×284, whole | 1.250 | 0.734 | 0.746 / 0.742 | 97.71 / 93.81 |
| Same jewelry | 380×284, 128px tiles | 2.822 | 2.624 | 2.626 / 2.531 | 97.70 / 93.87 |
| Native jewelry, 6 frames | 995×746, 512px tiles | 8.616 | 8.205 | 8.985 / 7.882 | 96.78 / 89.43 |
| Keyboard, 32 frames | 314×174, whole | 2.667 | 2.130 | 2.203 / 2.333 | 74.96 / 84.83 |

These are single CPU runs with diagnostics enabled for Conservative, while other
tests were running. Soft also incurs the first-session startup cost. They are
observations, not a controlled speed ranking. Process peak RSS was approximately
567 MiB for the small whole-frame jewelry run, 226 MiB tiled, 1,075 MiB native
jewelry and 1,149 MiB keyboard. These are cumulative process high-water readings,
including the model, inputs and diagnostics; they do not isolate policy overhead.
The extra admitted scoring scratch is bounded by
`min(height,280) * width * 160` bytes (about 21.9 MiB for a 512px-wide tile),
independent of the number of color source frames. No extra full N-frame color
copy is introduced.

At native jewelry resolution, default first/last selections had 45/6 override
components. The explicitly chosen background crop had 0% overrides and zero
source-label changes. These numbers do not establish perceptual improvement:
the original background was already fairly smooth. The ring-edge crop recovered
more detail at looser settings; stricter settings visibly kept blurred gold-band
edges or facets. The keyboard still loses or interrupts some sharp key lettering,
especially with first fallback. This remains a known failure case even with the
revised confidence score.

On the same small jewelry stack, tiled versus whole default Conservative differed
at 0.057% of pixels with first fallback and 0.727% with last. Mean absolute channel
differences were 0.0032 and 0.0986 on an 8-bit scale; maxima were 25 and 126.
Thus small average disagreement must not be mistaken for a bound on local visual
artifacts. Fixed-probability synthetic whole/tiled tests are exactly equal;
real model context can change decisions. The inspected sheets showed no obvious
rectangular tile joins, but this is not a guarantee for other stacks.

## Reproduce and remaining limitations

See [the comparison commands and exact policy](CONSERVATIVE_SELECTION.md).
For the native bundled example, the completed run used equivalent prepared
inputs from:

```sh
$PY tools/compare_conservative.py --input assets/demo_stack --align \
  --max-size 0 --tile-size 512 --output /tmp/v3-native --diagnostics \
  --crop ring-edge,400,200,128,128 --crop facets,450,400,128,128 \
  --crop overlap,660,390,128,128 --crop background,80,100,128,128
$PY tools/compare_conservative.py --input assets/Examples --align \
  --max-size 320 --output /tmp/v3-keyboard --diagnostics
```

Missing clean background cannot be reconstructed by source selection. Broad
smooth interiors, weak/ambiguous detail, noise, residual misalignment/exposure
changes and tile-context disagreement remain risks. Thresholds are normalized
for bit depth but not invariant to resampling or magnification. Use valid aligned
regions, inspect both fallback choices, and compare at native resolution. No
claim is made that this experiment already meets the ultimate mushroom-halo
quality goal; representative mushroom stacks and human review remain necessary.

## Email-ready summary (not sent)

Subject: OpenFocus V3 Conservative Selection implemented — tests and comparison results

Added opt-in Conservative Selection with first/last processed-frame fallback,
confidence and sharpness controls, exact source-pixel selection, deterministic
tile overlaps, 16-bit preservation, GUI/CLI/batch integration and diagnostic
comparisons. Defaults are first fallback, confidence 0.40 and sharpness advantage
0.25. Soft remains the default; model weights and Auto Background are unchanged.

Validation: the full suite passed 2,979 tests with 13 skips; targeted GUI/CLI,
policy and localization tests also passed. After separate storage/worker edits
appeared in the shared tree, the V3/conservative/streaming check passed 78 tests. Soft and Hard outputs matched the
original checkout byte-for-byte in whole-frame and tiled comparisons.

Jewelry and keyboard comparisons are available with Auto Background off. The
native jewelry background crop stayed entirely in fallback. Conservative still
misses some sharp keyboard lettering, and stricter settings lose jewelry detail.
Real mushroom validation is pending; background spill/occlusion and tile-context
changes remain limitations. Conservative depth export is explicitly unsupported.
The feature branch and reviewable diff are local; nothing was committed or sent.
