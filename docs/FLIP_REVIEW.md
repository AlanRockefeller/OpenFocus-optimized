# Flip review viewer

The flip review viewer compares renders of the same focus stack (different
methods, settings or code revisions) in one shared view and records ratings
and notes next to the images. It is a standalone development tool for now;
the viewer is a self-contained widget so the application can open it later.

```
python tools/flip_review.py before.tif after.tif [more.tif ...]
python tools/flip_review.py renders/            # every image in the folder
python tools/flip_review.py review-set.json     # a review-set manifest
```

Before any window opens, the command line reads every file header and
exits with status 2 and a message naming the files when an image cannot be
read, the pixel dimensions differ, or an existing review record does not
match the images (`--new-review` keeps that record as a backup and starts a
new one; `--review PATH` chooses another). A decode failure that the header
check could not foresee also ends the program with an error instead of
leaving an empty window. A single image opens without flipping. TIFF (including LZW), PNG and JPEG in 8 or 16 bits, grey
or colour, are decoded with the application's decoder and shown like the
main window shows them (16-bit samples divided by 257).

Options: `--review PATH` (record location), `--new-review` (keep a
mismatching record as a backup and start a new one), `--title`, `--blind`,
`--reviewer NAME`, `--theme dark|light`, `--fullscreen`, `--memory-mb MB`,
and `--export OUT.json|OUT.md` with `--export-paths` / `--export-reviewer`.

## Viewing

Zoom and pan are shared by every candidate: flipping swaps only the picture,
so the same pixels stay under the eye. The zoom is defined in physical screen pixels: 100 % shows one image
pixel per device pixel, and 200, 400 and 800 % show exact 2x2, 4x4 and 8x8
device-pixel blocks, also on scaled displays (device pixel ratio 2, 1.5,
1.25 ...). The fit level is computed from the device-pixel size of the view.
The viewer follows the ratio of the screen it is on, including moves to
another screen. The zoom runs from fit to 800 % with exact steps at 100,
200, 400 and 800 %. Below 100 % the picture is smoothed at device resolution
from an area-averaged pyramid; from 100 % up every image pixel is drawn as a
sharp square. The candidate label, its position (`2 / 4`), the rating and
the zoom level are drawn at a fixed place and logical size on a dark chip,
legible over bright and dark content.

Every candidate has its own colour, shown as a stripe and border on the
label chip and as a swatch in the candidate list, so a flip is visible at a
glance even between similar names. In blind review the colours belong to
the neutral letters (A is always the first colour), so they reveal nothing;
after reveal each picture keeps the colour it had.

| Key | Action |
| --- | --- |
| Left / Right | previous / next candidate (wraps around) |
| Space | back to the previously viewed candidate (A/B toggle) |
| Ctrl+1 ... Ctrl+9 | show candidate k |
| 1 ... 5 | overall rating; the same key again clears it |
| B | mark as best of the set (one per set) |
| + / - , wheel | zoom (the wheel keeps the point under the cursor) |
| 0 | fit |
| Z, double-click | toggle fit and 100 % |
| drag | pan |
| F or F11 | fullscreen for the picture: the notes panel hides (N shows it); Esc leaves fullscreen and restores the panel as it was |
| H | hide or show labels and pin markers |
| D / Shift+D | amplified absolute difference to the reference / cycle the gain (x4, x16, x64) |
| R | use the current candidate as the difference reference |
| P, then click; or Ctrl+click | pin a note to that pixel |
| N | show or hide the notes panel |
| C | edit the comment of the current candidate |
| ? or F1 | key help |

The difference is computed from the decoded samples at full bit depth,
`|current - reference| x gain`, and reduced for zoomed-out display by the
2x2 maximum so isolated changes stay visible.

The notes panel holds the rating buttons, the best mark, a comment per
candidate, optional criterion ratings (sharpness, halos, seams or artifacts,
noise, colour; collapsed and optional), the pinned notes, a comment for the
whole set, the reviewer name, blind review and export.

### Memory

Each candidate is decoded once into an 8-bit display buffer and a pyramid of
half-size levels (about 1.33 times the 8-bit image: 81 MB for a 3908x5216
colour render). Flipping only switches the prepared buffer. Levels at 1/4
size and smaller stay in memory for every candidate; full and half size
levels are kept within a budget (default: a quarter of the available memory,
at most 2 GiB; `--memory-mb` overrides). The current candidate, the
previously viewed one, the neighbours and the difference reference are kept
first; any other candidate shows its preview at once and reloads in the
background. Eight 20-megapixel candidates fit in the default budget.

### Blind review

With `--blind` or the panel checkbox, labels become neutral letters
(`Candidate A`, `B`, ...) in a shuffled order, which is stored in the record
so it stays stable between sessions. Ratings and notes made while blind
attach to the true candidates and carry a `rated_blind` flag. **Reveal**
(click twice to confirm) shows the real labels and restores the original
order. An export always uses the real labels.

## Review record

The record is JSON (`"format": "openfocus/flip-review"`, `"version": 1`),
written atomically (temporary file, flush, replace) shortly after every
change and when the viewer closes. Default location:

- files: `openfocus-review-<sorted file stems>.json` beside the first image,
- folder: `openfocus-review.json` in the folder,
- manifest: `<manifest name>.review.json` beside the manifest.

Each candidate is identified by file name, byte size, SHA-256 and decoded
pixel dimensions. On opening, candidates are matched by content, so input
order and renaming do not matter. If any image changed, is missing or is
extra, the viewer says which, does not apply or save the record, and offers
to start a new record that keeps the old file as
`<name>.mismatch-<time>.json`.

```json
{
  "format": "openfocus/flip-review", "version": 1,
  "title": "", "reviewer": "",
  "created": "2026-11-11T12:00:00Z", "modified": "2026-11-11T12:05:00Z",
  "tool": {"name": "OpenFocus flip review", "version": "1.77.0"},
  "criteria": ["sharpness", "halos", "seams", "noise", "colour"],
  "candidates": [
    {"id": "c1", "label": "before.tif",
     "file": {"name": "before.tif", "size": 122307020, "sha256": "5f92...",
              "width": 3908, "height": 5216},
     "path": "before.tif", "metadata": {},
     "rating": 4, "criteria": {"halos": 3}, "comment": "", "rated_blind": false}
  ],
  "reference": null, "best": "c1", "set_comment": "",
  "pins": [{"id": "p1", "x": 2310, "y": 2605, "candidate": "c1",
            "text": "halo at the stem", "created": "2026-11-11T12:03:00Z"}],
  "next_pin": 2,
  "blind": {"enabled": false, "revealed": false, "order": []}
}
```

`path` is relative to the record when the image is nearby. Pins are image
pixel coordinates (column `x`, row `y`), belong to the set and name the
candidate they describe.

## Export for sharing

**Export review...** in the panel, or `--export OUT`, writes a summary to
attach to an issue: JSON (`"format": "openfocus/flip-review-export"`) or,
for a `.md` name, a Markdown table. By default it contains the title,
labels, file names, sizes, SHA-256 and dimensions, metadata, ratings,
criteria, comments, pins and the set comment, and **no pixels, no file paths
and no reviewer name**. Absolute paths inside metadata are reduced to their
file name, and the home folder in any text is replaced by `~`. "Include file
paths" (`--export-paths`) and "Include reviewer name" (`--export-reviewer`)
add them explicitly. The tool has no network code.

## Review-set manifest

A runner can hand over a set as JSON. Paths are relative to the manifest.

```json
{
  "format": "openfocus/review-set",
  "version": 1,
  "title": "Tile overlap 32 vs 64",
  "reference": "overlap 32",
  "candidates": [
    {"label": "overlap 32", "path": "renders/before.tif",
     "metadata": {"revision": "3949463", "method": "openfocus_b",
                  "parameters": {"tile_overlap": 32}, "timings": {"render_s": 41.2}}},
    {"label": "overlap 64", "path": "renders/after.tif"}
  ]
}
```

- `candidates` (required, non-empty): objects with `path` (required),
  `label` (default: the file name; labels must be unique) and `metadata`
  (any JSON object, shown in the record and the export). A bare string is a
  path.
- `title`, `reference` (a label or a 0-based index; the default difference
  reference), `format` and `version` are optional; a newer `version` is
  refused.

## Embedding

- `core/flip_review.py`: review sets, records, matching and export; standard
  library only, no Qt.
- `widgets/flip_review.py`: `FlipReviewViewer(review_set, record_path,
  parent=...)`, a `QWidget` with signals `loaded`, `loadFailed`,
  `recordMismatch`, `saved`, `saveFailed` and `currentChanged`; call
  `shutdown()` before discarding it.
- `widgets/flip_review_text.py`: every user-visible English string, ready to
  move into the translation tables.
- `tools/flip_review.py`: the command line.
