"""Sample focus stacks kept outside OpenFocus: list, fetch, verify, locate and pack.

The stacks are published separately; OpenFocus does not ship them or a default
manifest. Name a manifest (a JSON file, a file:// URL or an https URL) with
--manifest or the OPENFOCUS_SAMPLE_MANIFEST environment variable. Downloads go
to a per-user cache; override it with --cache-dir or OPENFOCUS_SAMPLE_CACHE.
Format and publishing workflow: docs/SAMPLE_STACKS.md.

Commands:
  list       stacks and variants in the manifest, with local availability
  info ID    everything the manifest records about one stack
  fetch ID   download, verify and unpack into the cache (no-op when present)
  verify ID  re-hash every frame of a local or cached stack
  path ID    the stack's folder, or with --files its ordered frame paths
  check      validate the manifest (--strict also fails on placeholders)
  metadata   report GPS data, serial numbers and names in image files
  pack       hash files into a manifest entry and write publishable tar parts
  cache-dir  print the cache folder in use

Examples:
  python tools/sample_stacks.py --manifest stacks.json list
  python tools/sample_stacks.py --manifest stacks.json fetch my-stack
  python main.py -i $(python tools/sample_stacks.py path my-stack) -o out.tif
  python tools/sample_stacks.py path my-stack --files > frames.txt
  python tools/sample_stacks.py pack --id my-stack --folder ./photos --out ./release
"""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import sample_stacks as samples  # noqa: E402


def _human(size):
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


class _Progress:
    def __init__(self, quiet):
        self.quiet = quiet
        self.started = {}

    def __call__(self, label, done, total):
        if self.quiet:
            return
        start = self.started.setdefault(label, time.monotonic())
        rate = done / max(time.monotonic() - start, 1e-6)
        text = f"{label}: {_human(done)} of {_human(total)} ({_human(rate)}/s)"
        if sys.stderr.isatty():
            print("\r" + text.ljust(78), end="" if done < total else "\n",
                  file=sys.stderr, flush=True)
        elif done == total:
            print(text, file=sys.stderr, flush=True)


def _manifest(args):
    manifest = samples.load_manifest(args.manifest, cache_dir=args.cache_dir,
                                     offline=args.offline)
    if manifest.warnings and not args.quiet:
        print(f"note: the manifest has {len(manifest.warnings)} warning(s); "
              "the check command lists them", file=sys.stderr)
    return manifest


def _status(stack, manifest, args):
    found = samples.available_folder(stack, manifest, args.cache_dir)
    return found[1] if found else "not fetched"


def cmd_list(args):
    manifest = _manifest(args)
    rows = []
    for stack in manifest.stacks.values():
        tags = stack.data.get("tags") or []
        if args.tag and args.tag not in tags:
            continue
        rows.append({"id": stack.id, "title": stack.title, "frames": len(stack.frames),
                     "format": stack.data.get("format"), "bytes": stack.total_bytes,
                     "status": _status(stack, manifest, args), "variant_of": stack.parent,
                     "tags": tags})
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("(no stacks)")
    for row in rows:
        name = ("  " + row["id"]) if row["variant_of"] else row["id"]
        print(f"{name:<40} {row['frames']:>4} {row['format'] or '':<5} "
              f"{_human(row['bytes']):>10}  {row['status']:<11}  {row['title']}")
    return 0


def cmd_info(args):
    manifest = _manifest(args)
    stack = manifest.get(args.id)
    summary = samples.stack_summary(stack)
    summary["status"] = _status(stack, manifest, args)
    summary["cache_folder"] = str(samples.cache_root(args.cache_dir) / "stacks" / stack.id
                                  / stack.content_key)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    print(f"{stack.id}: {stack.title}")
    if stack.parent:
        print(f"  variant ({stack.kind}) of {stack.parent}")
    print(f"  {summary.get('description', '')}")
    print(f"  frames: {len(stack.frames)} {summary.get('format')} "
          f"{summary.get('width')}x{summary.get('height')} {summary.get('bit_depth')}-bit, "
          f"{_human(stack.total_bytes)} ({stack.total_bytes} bytes)")
    print(f"  useful for: {'; '.join(summary.get('useful_for') or []) or '-'}")
    print(f"  tags: {', '.join(summary.get('tags') or []) or '-'}")
    print(f"  licence: {summary.get('license')}")
    print(f"  attribution: {summary.get('attribution')}")
    if summary.get("recommended"):
        print(f"  recommended: {json.dumps(summary['recommended'], ensure_ascii=False)}")
    for archive in stack.archives:
        print(f"  archive: {archive.name} {_human(archive.bytes)} sha256 {archive.sha256}")
        print(f"           {archive.url}")
    if stack.local_path:
        print(f"  local_path: {stack.local_path}")
    if stack.variant_ids:
        print(f"  variants: {', '.join(stack.variant_ids)}")
    print(f"  status: {summary['status']}")
    print(f"  first/last frame: {stack.frames[0].path} ... {stack.frames[-1].path}")
    return 0


def cmd_fetch(args):
    manifest = _manifest(args)
    status = 0
    for stack_id in args.ids:
        stack = manifest.get(stack_id)
        started = time.monotonic()
        try:
            folder = samples.fetch_stack(stack, manifest, cache_dir=args.cache_dir,
                                         offline=args.offline, keep_downloads=args.keep_downloads,
                                         retries=args.retries, progress=_Progress(args.quiet))
        except samples.SampleStackError as exc:
            print(f"error: {stack_id}: {exc}", file=sys.stderr)
            status = 1
            continue
        print(f"{stack_id}: {folder} ({len(stack.frames)} frames, "
              f"{time.monotonic() - started:.1f} s)")
    return status


def cmd_verify(args):
    manifest = _manifest(args)
    status = 0
    for stack_id in args.ids:
        stack = manifest.get(stack_id)
        started = time.monotonic()
        result = samples.verify_stack(stack, manifest, cache_dir=args.cache_dir)
        if result.ok:
            print(f"{stack_id}: OK, {result.files} frames, {_human(result.bytes)} "
                  f"({result.source}: {result.folder}, {time.monotonic() - started:.1f} s)")
            continue
        status = 1
        print(f"{stack_id}: FAILED", file=sys.stderr)
        for problem in result.problems:
            print(f"  {problem}", file=sys.stderr)
    return status


def cmd_path(args):
    manifest = _manifest(args)
    stack = manifest.get(args.id)
    files = samples.stack_files(args.id, manifest, cache_dir=args.cache_dir,
                                fetch=not args.no_fetch, offline=args.offline,
                                progress=_Progress(args.quiet))
    if args.files or args.shell or args.null:
        if args.null:
            sys.stdout.write("".join(f"{path}\0" for path in files))
        elif args.shell:
            print(" ".join(shlex.quote(str(path)) for path in files))
        else:
            print("\n".join(str(path) for path in files))
        return 0
    folder = files[0].parent
    exact, reason = samples.folder_is_exact(folder, stack)
    if not exact:
        print(f"error: {folder} is not exactly stack {stack.id!r} ({reason}); "
              "use --files (or --shell) to get the ordered frame list", file=sys.stderr)
        return 1
    print(folder)
    return 0


def cmd_check(args):
    location = samples.manifest_source(args.manifest)
    try:
        manifest = samples.load_manifest(location, cache_dir=args.cache_dir, offline=args.offline)
        issues = manifest.issues
    except samples.ManifestError as exc:
        issues = exc.issues or [samples.Issue("error", "manifest", str(exc))]
        manifest = None
    for issue in issues:
        print(issue)
    errors = sum(issue.level == "error" for issue in issues)
    warnings = len(issues) - errors
    if manifest is not None and args.files:
        for stack in manifest.stacks.values():
            if stack.local_path is not None:
                missing = [f.path for f in stack.frames
                           if not (stack.local_path / f.path).is_file()]
                if missing:
                    errors += 1
                    print(f"error: {stack.id}: {len(missing)} frame(s) missing in "
                          f"{stack.local_path}, e.g. {missing[:3]}")
    count = len(manifest.stacks) if manifest else 0
    print(f"{location}: {count} stack(s) and variant(s), {errors} error(s), {warnings} warning(s)")
    if errors or (args.strict and warnings):
        return 1
    return 0


def _input_files(args, parser):
    sources = [bool(args.folder), bool(args.files_from), bool(args.files)]
    if sum(sources) != 1:
        parser.error("give exactly one of --folder, --files-from or file arguments")
    if args.folder:
        files = samples.list_stack_files(args.folder)
        if not files:
            parser.error(f"no image files in {args.folder}")
        return files
    if args.files_from:
        lines = Path(args.files_from).read_text(encoding="utf-8").splitlines()
        return [Path(line.strip()) for line in lines if line.strip() and not line.startswith("#")]
    return [Path(f) for f in args.files]


def cmd_metadata(args, parser):
    files = []
    for item in args.files:
        path = Path(item)
        files.extend(samples.list_stack_files(path) if path.is_dir() else [path])
    if args.files_from:
        files.extend(Path(line.strip()) for line in
                     Path(args.files_from).read_text(encoding="utf-8").splitlines() if line.strip())
    if not files:
        parser.error("give image files, folders or --files-from")
    report = samples.inspect_metadata(files, use_exiftool=not args.no_exiftool)
    if args.json:
        print(json.dumps(report.as_dict(show_values=not args.hide_names), indent=2,
                         ensure_ascii=False))
    else:
        print("\n".join(report.summary_lines(show_values=not args.hide_names)))
        if report.blocking:
            print("pack refuses these files without --allow-identifying-metadata")
    return 3 if report.blocking else 0


def cmd_pack(args, parser):
    files = _input_files(args, parser)
    if not args.local_only and not args.out and not args.variant_of:
        parser.error("--out is required unless --local-only or --variant-of is given")
    say = (lambda message: None) if args.quiet else (lambda message: print(message, file=sys.stderr))
    target = Path(args.add_to) if args.add_to else None
    if args.variant_of:
        if target is None:
            parser.error("--variant-of needs --add-to MANIFEST holding the parent stack")
        data = json.loads(target.read_text(encoding="utf-8"))
        parent = next((s for s in data.get("stacks", []) if s.get("id") == args.variant_of), None)
        if parent is None:
            parser.error(f"{target} has no stack {args.variant_of!r}")
        say(f"Checking {len(files)} file(s) against {args.variant_of}...")
        entry = samples.subset_variant(parent, files, args.id, title=args.title,
                                       description=args.description,
                                       useful_for=args.useful_for, tags=args.tag,
                                       recommended=_recommended(args))
        report = None
    else:
        recommended = _recommended(args)
        try:
            entry, report = samples.pack_stack(
                files, args.id, out_dir=args.out, local_only=args.local_only,
                title=args.title, description=args.description, useful_for=args.useful_for,
                tags=args.tag, license=args.license, attribution=args.attribution,
                base_url=args.base_url, part_bytes=args.part_size, file_format=args.format,
                width=args.width, height=args.height, bit_depth=args.bit_depth,
                recommended=recommended,
                allow_identifying_metadata=args.allow_identifying_metadata,
                use_exiftool=not args.no_exiftool, progress=say)
        except samples.IdentifyingMetadataError as exc:
            print("\n".join(exc.report.summary_lines()), file=sys.stderr)
            print(f"error: {exc}", file=sys.stderr)
            return 3
        print("\n".join(report.summary_lines()), file=sys.stderr)
    text = json.dumps(entry, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        samples.write_json_atomic(out / f"{args.id}.entry.json", entry)
        say(f"Wrote {out / (args.id + '.entry.json')}")
    if target is not None:
        data = (json.loads(target.read_text(encoding="utf-8")) if target.exists()
                else samples.empty_manifest())
        samples.merge_entry(data, entry, variant_of=args.variant_of)
        issues = samples.validate_manifest(data)
        errors = [i for i in issues if i.level == "error"]
        if errors:
            for issue in errors:
                print(issue, file=sys.stderr)
            print(f"error: {target} was not changed", file=sys.stderr)
            return 1
        samples.write_json_atomic(target, data)
        say(f"Updated {target}")
    if not args.quiet:
        print(text)
    return 0


def _recommended(args):
    recommended = {}
    if args.recommend_method:
        recommended["method"] = args.recommend_method
    if args.recommend_alignment:
        recommended["alignment"] = args.recommend_alignment
    if args.recommend_load_scale:
        recommended["load_scale"] = args.recommend_load_scale
    if args.recommend_args:
        recommended["cli_args"] = shlex.split(args.recommend_args)
    return recommended or None


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", default=argparse.SUPPRESS,
                        help=f"manifest path or URL (default: ${samples.MANIFEST_ENV})")
    common.add_argument("--cache-dir", default=argparse.SUPPRESS,
                        help=f"download cache folder (default: ${samples.CACHE_ENV} or the "
                             "per-user cache)")
    common.add_argument("--offline", action="store_true", default=argparse.SUPPRESS,
                        help="never use the network; use only cached data")
    common.add_argument("--quiet", "-q", action="store_true", default=argparse.SUPPRESS,
                        help="no progress output")
    parser = argparse.ArgumentParser(description=__doc__, parents=[common],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    item = commands.add_parser("list", parents=[common], help="list stacks")
    item.add_argument("--tag", help="only stacks with this tag")
    item.add_argument("--json", action="store_true")
    item = commands.add_parser("info", parents=[common], help="describe one stack")
    item.add_argument("id")
    item.add_argument("--json", action="store_true")
    item = commands.add_parser("fetch", parents=[common], help="download and verify stacks")
    item.add_argument("ids", nargs="+", metavar="ID")
    item.add_argument("--keep-downloads", action="store_true",
                      help="keep verified archives after unpacking (uses twice the space)")
    item.add_argument("--retries", type=int, default=3, help="network retries (default 3)")
    item = commands.add_parser("verify", parents=[common], help="re-hash local or cached stacks")
    item.add_argument("ids", nargs="+", metavar="ID")
    item = commands.add_parser("path", parents=[common],
                               help="print the stack folder or ordered frame paths")
    item.add_argument("id")
    item.add_argument("--files", action="store_true", help="one frame path per line, in order")
    item.add_argument("--shell", action="store_true",
                      help="frame paths on one line, shell-quoted, for main.py -i")
    item.add_argument("--null", action="store_true", help="NUL-separated frame paths")
    item.add_argument("--no-fetch", action="store_true", help="fail instead of downloading")
    item = commands.add_parser("check", parents=[common], help="validate the manifest")
    item.add_argument("--strict", action="store_true",
                      help="also fail on warnings such as licence placeholders")
    item.add_argument("--files", action="store_true",
                      help="also check that local_path frames exist")
    item = commands.add_parser("metadata", parents=[common],
                               help="report identifying metadata in image files")
    item.add_argument("files", nargs="*", metavar="FILE_OR_FOLDER")
    item.add_argument("--files-from", metavar="LIST", help="text file with one path per line")
    item.add_argument("--no-exiftool", action="store_true", help="use only the built-in reader")
    item.add_argument("--hide-names", action="store_true", help="do not print name values")
    item.add_argument("--json", action="store_true")
    item = commands.add_parser("pack", parents=[common],
                               help="make archive parts and a manifest entry")
    item.add_argument("files", nargs="*", metavar="FILE", help="frames in stack order")
    item.add_argument("--folder", help="pack the image files of this folder in name order")
    item.add_argument("--files-from", metavar="LIST", help="text file with one path per line, in order")
    item.add_argument("--id", required=True, help="stack id (a-z, 0-9, '.', '_', '-')")
    item.add_argument("--out", help="folder for the tar parts and <id>.entry.json")
    item.add_argument("--local-only", action="store_true",
                      help="write an entry that points at the files' folder; no archives")
    item.add_argument("--variant-of", metavar="PARENT",
                      help="record the files as a subset variant of PARENT (in --add-to)")
    item.add_argument("--add-to", metavar="MANIFEST",
                      help="insert or replace the entry in this manifest file (created if missing)")
    item.add_argument("--base-url", help="URL prefix of the published assets, e.g. "
                                         "https://github.com/OWNER/REPO/releases/download/TAG")
    item.add_argument("--part-size", type=int, default=samples.DEFAULT_PART_BYTES,
                      help=f"maximum bytes per tar part (default {samples.DEFAULT_PART_BYTES})")
    item.add_argument("--title")
    item.add_argument("--description", default="")
    item.add_argument("--useful-for", action="append", default=[], metavar="TEXT")
    item.add_argument("--tag", action="append", default=[])
    item.add_argument("--license", default=samples.PLACEHOLDER_PREFIX
                      + ": choose an SPDX licence identifier",
                      help="SPDX identifier, e.g. CC-BY-4.0")
    item.add_argument("--attribution", default=samples.PLACEHOLDER_PREFIX
                      + ": photographer name and credit line")
    item.add_argument("--format", help="file format label (default: from the extension)")
    item.add_argument("--width", type=int)
    item.add_argument("--height", type=int)
    item.add_argument("--bit-depth", type=int)
    item.add_argument("--recommend-method", metavar="METHOD")
    item.add_argument("--recommend-alignment", metavar="ALIGN")
    item.add_argument("--recommend-load-scale", type=float, metavar="SCALE")
    item.add_argument("--recommend-args", metavar="ARGS",
                      help="extra openfocus CLI arguments, as one shell-quoted string")
    item.add_argument("--allow-identifying-metadata", action="store_true",
                      help="pack even though GPS data or serial numbers were found")
    item.add_argument("--no-exiftool", action="store_true", help="use only the built-in reader")
    commands.add_parser("cache-dir", parents=[common], help="print the cache folder")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, default in (("manifest", None), ("cache_dir", None), ("offline", False),
                          ("quiet", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    try:
        if args.command == "cache-dir":
            print(samples.cache_root(args.cache_dir))
            return 0
        if args.command == "metadata":
            return cmd_metadata(args, parser)
        if args.command == "pack":
            return cmd_pack(args, parser)
        return {"list": cmd_list, "info": cmd_info, "fetch": cmd_fetch, "verify": cmd_verify,
                "path": cmd_path, "check": cmd_check}[args.command](args)
    except samples.SampleStackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, samples.ManifestNotConfigured) else 1
    except KeyboardInterrupt:
        print("interrupted; partial downloads are kept and resume on the next fetch",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
