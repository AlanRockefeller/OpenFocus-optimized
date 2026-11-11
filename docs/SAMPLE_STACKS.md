# Sample stacks

People who change a fusion or registration method, or compare methods, need
the same real focus stacks to test on. The stacks are large (camera RAW
stacks are often 0.3 to 1.6 GB), so they are never part of an OpenFocus
download or of this repository. They are published separately, as release
assets of a data-only repository, and described by a JSON **manifest**.

OpenFocus carries only the tooling:

- `utils/sample_stacks.py`: manifest validation, the download cache and the
  packer. Standard library only; no GUI or numerical imports.
- `tools/sample_stacks.py`: the command line.

OpenFocus does not ship a manifest. Name one with `--manifest PATH_OR_URL` or
the `OPENFOCUS_SAMPLE_MANIFEST` environment variable. A manifest can be a local
file, a `file://` URL or an `https://` URL.

## Using a stack

```bash
export OPENFOCUS_SAMPLE_MANIFEST=/path/to/manifest.json   # or pass --manifest

python tools/sample_stacks.py list                 # stacks, variants, availability
python tools/sample_stacks.py info STACK_ID        # description, size, licence
python tools/sample_stacks.py fetch STACK_ID       # download, verify, unpack
python tools/sample_stacks.py verify STACK_ID      # re-hash every frame
python tools/sample_stacks.py path STACK_ID        # folder holding exactly the stack
python tools/sample_stacks.py path STACK_ID --files   # ordered frame paths, one per line
python tools/sample_stacks.py check                # validate the manifest
```

`path` fetches the stack first when needed (`--no-fetch` fails instead).
Without `--files` it prints the folder only when loading that folder gives
exactly the stack's frames in the recorded order; otherwise (a local folder
with other photos, a subset variant, an order that differs from the name
order) it fails and asks for `--files`. To fuse a stack:

```bash
python main.py -i "$(python tools/sample_stacks.py path STACK_ID)" -o fused.tif
# Any stack, including subsets of a larger folder (paths without spaces):
python main.py -i $(python tools/sample_stacks.py path STACK_ID --files) -o fused.tif
# Any paths, with bash:
mapfile -t frames < <(python tools/sample_stacks.py path STACK_ID --files)
python main.py -i "${frames[@]}" -o fused.tif
```

`--shell` prints the paths shell-quoted on one line, for pasting into a
command; `--null` prints NUL-separated paths for `xargs -0`.

From Python (benchmarks and tests):

```python
from utils.sample_stacks import stack_files

paths = stack_files("STACK_ID")                    # $OPENFOCUS_SAMPLE_MANIFEST
paths = stack_files("STACK_ID", "/path/manifest.json", offline=True)
```

`stack_files` returns the ordered frame paths, fetching and verifying the
stack on first use. `fetch=False` raises instead of downloading.

Exit status: 0 success; 1 failure (verification, download, manifest errors);
2 usage error or no manifest configured; 3 identifying metadata found
(`metadata` and `pack`).

## Cache

Fetched stacks live in a per-user cache folder (`cache-dir` prints it):

| Condition | Folder |
|---|---|
| `--cache-dir DIR` | `DIR` |
| `OPENFOCUS_SAMPLE_CACHE` set | that folder |
| `OPENFOCUS_DATA_DIR` set, or a portable installation | `<OpenFocus data folder>/sample-stacks` |
| Windows | `%LOCALAPPDATA%\OpenFocus\Cache\sample-stacks` |
| macOS | `~/Library/Caches/OpenFocus/sample-stacks` |
| Linux and others | `$XDG_CACHE_HOME/OpenFocus/sample-stacks` (default `~/.cache/...`) |

Layout:

- `downloads/<sha256>.part`: an unfinished download. The next fetch resumes
  it with an HTTP range request, or starts again when the server cannot
  resume. A resumed file whose digest is wrong is downloaded once more from
  the start.
- `downloads/<sha256>`: a verified archive. It is renamed into place only
  after its size and SHA-256 match the manifest, and removed after unpacking
  unless `fetch --keep-downloads` is given (keeping it doubles the space used).
- `stacks/<id>/<key>/`: an unpacked stack. `<key>` is derived from the ordered
  frame list, so a changed stack gets a new folder. The stack is unpacked into
  `stacks/<id>/.tmp-*`, every frame's size and SHA-256 is checked, the marker
  `.openfocus-sample-stack.json` is written, and only then is the folder
  renamed into place. A folder without a matching marker is never used, and
  leftovers of an interrupted fetch are removed by the next fetch of that
  stack.

A size or digest mismatch fails with an error and leaves no partial stack.
Once fetched, a stack is used without network access; `--offline` never
touches the network. A remote manifest is cached too and used offline.

Archives are unpacked member by member. Absolute names, `..`, drive letters,
backslashes, links and device entries reject the whole archive; only the
frames the manifest lists are written, and only inside the target folder.
Uncompressed and compressed tar files and zip files are accepted.

Remove a stack by deleting its folder under `stacks/`, or the whole cache.

## Manifest format

Schema identifier: `openfocus-sample-stacks/1`. Unknown keys produce
warnings; keys starting with `x-` are free for extensions.

```json
{
  "schema": "openfocus-sample-stacks/1",
  "title": "OpenFocus sample stacks",
  "base_url": "https://github.com/OWNER/REPO/releases/download/v1/",
  "stacks": [
    {
      "id": "fungal-stalks-52",
      "title": "Translucent fungal stalks, 52 RAW frames",
      "description": "Subject, equipment, lighting, how the frames were taken.",
      "useful_for": ["many frames", "fine hairs against a dark background"],
      "tags": ["raw", "macro", "dark-background"],
      "frame_count": 52,
      "format": "ORF",
      "width": 5220,
      "height": 3912,
      "bit_depth": 12,
      "total_bytes": 919002195,
      "license": "CC-BY-4.0",
      "attribution": "Photo: Name Surname",
      "recommended": {
        "alignment": "openfocus",
        "method": "openfocus_b",
        "parameters": {"radius": 8},
        "align_parameters": {},
        "load_scale": 0.5,
        "cli_args": ["--native-raw"],
        "notes": "why these settings"
      },
      "archives": [
        {"name": "fungal-stalks-52.tar", "url": "fungal-stalks-52.tar",
         "bytes": 919040000, "sha256": "64 hex digits"}
      ],
      "frames": [
        {"path": "PB054292.ORF", "bytes": 17841127, "sha256": "64 hex digits", "archive": 0}
      ],
      "variants": [
        {"id": "fungal-stalks-52-first12", "kind": "subset",
         "title": "First 12 frames", "description": "Quick runs.",
         "frames": ["PB054292.ORF", "..."]}
      ]
    }
  ]
}
```

Stack fields (all required unless marked optional):

| Field | Meaning |
|---|---|
| `id` | 1-80 characters of `a-z 0-9 . _ -`, unique across stacks and variants |
| `title`, `description` | Short name; subject, equipment and capture notes |
| `useful_for` | What the stack tests well: many frames, fine hairs against a dark background, low contrast, large tiles, ... |
| `tags` | Short keywords for `list --tag` |
| `frame_count`, `total_bytes` | Must match `frames` |
| `format` | File format of the frames, e.g. `ORF`, `JPEG`, `TIFF` |
| `width`, `height`, `bit_depth` | Pixel dimensions and bits per sample of the recorded data (for RAW: the sensor data) |
| `license` | SPDX identifier or expression, e.g. `CC-BY-4.0`. A value starting with `TODO` is a placeholder: a warning, an error with `check --strict` |
| `attribution` | Credit line required by the licence |
| `frames` | The frame order: `path` (relative, `/`-separated), `bytes`, `sha256`, and how to obtain it |
| `archives` | Optional. Each `url`, `bytes` (below 2 GiB), `sha256`, optional `name` |
| `local_path` | Optional. Folder holding the frames on this machine; relative paths are relative to the manifest file |
| `recommended` | Optional. `alignment`, `method`, `parameters`, `align_parameters`, `load_scale` (0 to 1], `cli_args`, `notes` |
| `variants` | Optional. Reduced versions, see below |
| `references` | Optional, reserved (see below) |

Each frame is obtained from the archive with index `archive`, from its own
`url` (loose files), or from the stack's `local_path`. When `local_path`
exists and holds every frame with the listed size, it is used directly and
nothing is copied or downloaded; otherwise the archives or URLs are used.
A manifest of local entries therefore works before anything is published.

URLs may be absolute (`https://`, `file://`) or relative. Relative URLs are
resolved against `base_url`, or, without one, against the manifest's own
location, so a manifest saved next to its archives works from a folder too.

A stack larger than one asset is split into several self-contained tar parts,
each below 2 GiB; every frame names the part holding it. Fetching a stack
downloads only the parts its frames need.

### Variants

`variants` lists reduced versions of a stack. Each has an `id`, `kind`,
`title` and `description`, and is addressed by its `id` like a stack.

- `kind: "subset"`: `frames` lists parent frame paths (for example the first
  12). Everything else, including archives and `local_path`, comes from the
  parent. When the parent is already fetched its files are used; otherwise
  only the archive parts holding the subset are downloaded.
- `kind: "crop"`, `"downscale"` or `"other"`: different files. The variant has
  its own `frames`, `archives` or `local_path`, `format`, `width`, `height`,
  `bit_depth`, `frame_count` and `total_bytes`; `license`, `attribution`,
  `useful_for` and `tags` default to the parent's.

### Reserved for reference renders

`references` is reserved for reference renders and difference limits, so that
a later runner can check a candidate output against stored limits. Schema 1
only requires it to be a list when present; its contents will be defined by a
later schema version.

## Publishing a data repository

1. Choose the stacks and a licence for them (for example `CC-BY-4.0` or
   `CC0-1.0`) and an attribution line. Publish only photographs you own or
   may redistribute.
2. Check the metadata the files would expose:

   ```bash
   python tools/sample_stacks.py metadata /path/to/stack-folder
   python tools/sample_stacks.py metadata --files-from frames.txt --json
   ```

   It reports GPS data, camera/lens/accessory serial numbers, owner, artist,
   author and copyright names, and unique image identifiers (kinds and tag
   names; it never prints GPS or serial values). With `exiftool` installed
   it reads maker notes, XMP and IPTC as well; without it, the built-in reader
   checks standard EXIF/TIFF tags of JPEG and TIFF-based RAW files and says
   what it could not check. Pixel content (visible labels, text, faces) is
   never checked. The tools never modify photographs. If you want GPS
   positions or serial numbers removed, do that with your own tools on
   copies before packing, and run the check again.
3. Pack each stack. Files are stored unchanged in uncompressed tar parts
   (RAW files are never recompressed or rewritten), with fixed member order,
   times, owners and modes, so packing the same files again gives identical
   archives:

   ```bash
   python tools/sample_stacks.py pack --id fungal-stalks-52 \
       --files-from frames.txt --out release/ \
       --base-url https://github.com/OWNER/REPO/releases/download/v1 \
       --title "..." --description "..." --useful-for "many frames" --tag raw \
       --license CC-BY-4.0 --attribution "Photo: Name Surname" \
       --add-to release/manifest.json
   ```

   `--folder DIR` packs the image files of a folder in OpenFocus name order;
   `--files-from` and file arguments keep the given order. Parts default to
   at most 1,900,000,000 bytes (`--part-size`). Width, height and bit depth
   come from the first frame; `--width`, `--height` and `--bit-depth` override
   them (for RAW files give the sensor data bit depth, e.g. 12). The
   `recommend-*` options fill `recommended`. Pack refuses files with GPS data
   or serial numbers unless `--allow-identifying-metadata` is given.
   `release/<id>.entry.json` holds the entry; `--add-to` inserts or replaces
   it in a manifest. For a subset variant of a stack already in that
   manifest:

   ```bash
   python tools/sample_stacks.py pack --id fungal-stalks-52-first12 \
       --variant-of fungal-stalks-52 --files-from first12.txt \
       --add-to release/manifest.json --title "First 12 frames" --description "..."
   ```

4. Check the manifest: `python tools/sample_stacks.py --manifest
   release/manifest.json check --strict` (placeholders fail).
5. Create the data repository with a README stating the licence and
   attribution, and a release (for example `v1`). Upload the `.tar` parts as
   release assets (each must be below 2 GiB) and upload `manifest.json` as an
   asset too or commit it to the repository. Do not change an asset after
   publishing; publish a new release and manifest instead.
6. Test from a clean cache:
   `python tools/sample_stacks.py --manifest URL --cache-dir /tmp/x fetch ID`.

To try a manifest before publishing, keep it in the folder that holds the
parts (relative URLs) and fetch from there, or write a local-only entry:

```bash
python tools/sample_stacks.py pack --local-only --id my-stack \
    --files-from frames.txt --add-to ~/sample-stacks/manifest.local.json
```

A local-only entry points at the frames' folder and records their sizes and
SHA-256; nothing is copied. Its metadata report is printed but does not stop
it, because nothing is published.
