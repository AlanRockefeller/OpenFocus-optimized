"""Sample-stack manifest, download cache and packing, without network access.

Downloads are served by a local HTTP server thread or read from file URLs.
"""
import contextlib
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import threading
import zipfile

import pytest

from utils import sample_stacks as samples

from tools import sample_stacks as cli

ROOT = Path(__file__).resolve().parents[1]


# ---- helpers -------------------------------------------------------------------

def _frames(folder, count=5, size=3000, seed=1, suffix=".jpg"):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        data = hashlib.sha256(f"{seed}-{index}".encode()).digest() * (size // 32 + 1)
        path = folder / f"frame_{index + 1}{suffix}"
        path.write_bytes(data[:size + index])
        paths.append(path)
    return paths


def _pack(tmp_path, count=5, size=3000, part_bytes=samples.DEFAULT_PART_BYTES, **kwargs):
    files = _frames(tmp_path / "source", count, size)
    out = tmp_path / "release"
    entry, _report = samples.pack_stack(
        files, "demo", out_dir=out, title="Demo", description="Synthetic frames",
        useful_for=["tests"], tags=["synthetic"], license="CC0-1.0",
        attribution="OpenFocus tests", part_bytes=part_bytes, width=8, height=8,
        bit_depth=8, use_exiftool=False, **kwargs)
    return files, out, entry


def _write_manifest(path, entries, **extra):
    data = dict(samples.empty_manifest(), **extra)
    data["stacks"] = entries
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class _Server:
    """Threaded HTTP server with Range support and scripted failures."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.requests = []
        self.truncate_next = 0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                server.requests.append((self.path, self.headers.get("Range")))
                path = server.folder / self.path.lstrip("/")
                if not path.is_file():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                start = 0
                byte_range = self.headers.get("Range")
                if byte_range:
                    start = int(byte_range.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if server.truncate_next:
                    server.truncate_next -= 1
                    self.wfile.write(body[:len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    with contextlib.suppress(OSError):
                        self.connection.shutdown(2)
                    return
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(tmp_path):
    instance = _Server(tmp_path / "release")
    yield instance
    instance.close()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.delenv(samples.MANIFEST_ENV, raising=False)
    monkeypatch.setenv(samples.CACHE_ENV, str(tmp_path / "cache"))


def _http_manifest(tmp_path, server, entry):
    return samples.Manifest(dict(samples.empty_manifest(), base_url=server.url + "/",
                                 stacks=[entry]))


# ---- manifest ------------------------------------------------------------------

def test_missing_manifest_configuration_is_explained(capsys):
    with pytest.raises(samples.ManifestNotConfigured, match=samples.MANIFEST_ENV):
        samples.load_manifest()
    assert cli.main(["list"]) == 2
    assert "--manifest" in capsys.readouterr().err


def test_manifest_validation_reports_each_problem(tmp_path):
    _files, _out, entry = _pack(tmp_path)
    good = json.loads(json.dumps(entry))
    assert samples.validate_manifest(dict(samples.empty_manifest(), stacks=[good])) == []

    bad = json.loads(json.dumps(entry))
    bad["id"] = "Bad Id"
    bad["frame_count"] = 99
    bad["total_bytes"] = 1
    bad["frames"][1]["path"] = "../escape.jpg"
    bad["frames"][2]["sha256"] = "XYZ"
    bad["frames"][3]["archive"] = 7
    bad["archives"][0]["bytes"] = samples.MAX_ASSET_BYTES
    bad["license"] = "not a licence!"
    bad["recommended"] = {"load_scale": 3}
    bad["variants"] = [{"id": "sub", "kind": "subset", "title": "t", "description": "d",
                        "frames": ["missing.jpg"]}]
    messages = [str(i) for i in samples.validate_manifest(
        dict(samples.empty_manifest(), stacks=[bad, good, dict(good)]))]
    joined = "\n".join(messages)
    for expected in ("'id' 'Bad Id'", "'frame_count' is 99", "'total_bytes' is 1",
                     "unsafe or missing relative path '../escape.jpg'", "'sha256' must be 64",
                     "'archive' 7 is not an index", "2 GiB asset limit", "not an SPDX",
                     "load_scale", "frames not in the parent stack", "duplicate id 'demo'"):
        assert expected in joined, expected
    with pytest.raises(samples.ManifestError, match="error"):
        samples.Manifest(dict(samples.empty_manifest(), stacks=[bad]))
    assert "'schema' must be" in str(samples.validate_manifest({"stacks": []})[0])


def test_placeholders_are_warnings_and_strict_check_fails(tmp_path, capsys):
    files = _frames(tmp_path / "src", 2)
    with pytest.raises(samples.SampleStackError, match="--width"):
        samples.pack_stack(files, "local", local_only=True, use_exiftool=False)
    entry, _ = samples.pack_stack(files, "local", local_only=True, use_exiftool=False,
                                  useful_for=["x"], tags=["y"], width=4, height=4, bit_depth=8)
    path = _write_manifest(tmp_path / "m.json", [entry])
    assert cli.main(["--manifest", str(path), "check"]) == 0
    assert cli.main(["check", "--manifest", str(path), "--strict"]) == 1
    assert "'license' is a placeholder" in capsys.readouterr().out


# ---- packing and fetching --------------------------------------------------------

def test_pack_is_deterministic_and_stores_files_unchanged(tmp_path):
    files, out, entry = _pack(tmp_path)
    first = entry["archives"][0]["sha256"]
    again, _ = samples.pack_stack(files, "demo", out_dir=tmp_path / "again", use_exiftool=False,
                                  width=8, height=8, bit_depth=8)
    assert again["archives"][0]["sha256"] == first
    with tarfile.open(out / "demo.tar") as bundle:
        assert bundle.getnames() == [f.name for f in files]
        for path in files:
            assert bundle.extractfile(path.name).read() == path.read_bytes()
            info = bundle.getmember(path.name)
            assert (info.mtime, info.uid, info.mode) == (0, 0, 0o644)


def test_pack_then_fetch_round_trip_with_file_urls(tmp_path):
    files, out, entry = _pack(tmp_path)
    manifest_path = _write_manifest(out / "manifest.json", [entry])
    paths = samples.stack_files("demo", str(manifest_path))
    assert [p.name for p in paths] == [f.name for f in files]
    assert all(p.read_bytes() == f.read_bytes() for p, f in zip(paths, files))
    assert (paths[0].parent / samples.MARKER_NAME).is_file()
    assert samples.verify_stack(samples.load_manifest(manifest_path).get("demo")).ok
    assert not list((tmp_path / "cache" / "downloads").glob("*.part"))
    assert not (tmp_path / "cache" / "downloads" / entry["archives"][0]["sha256"]).exists()


def test_http_fetch_then_offline_reuse(tmp_path, server, capsys):
    _files, _out, entry = _pack(tmp_path)
    manifest_path = _write_manifest(tmp_path / "m.json", [entry], base_url=server.url)
    assert cli.main(["--manifest", str(manifest_path), "fetch", "demo", "-q"]) == 0
    assert len(server.requests) == 1
    server.close()
    files = samples.stack_files("demo", str(manifest_path), offline=True)
    assert len(files) == 5
    capsys.readouterr()
    assert cli.main(["--manifest", str(manifest_path), "--offline", "path", "demo"]) == 0
    assert capsys.readouterr().out.strip() == str(files[0].parent)
    assert cli.main(["--manifest", str(manifest_path), "path", "demo", "--files"]) == 0
    assert capsys.readouterr().out.split() == [str(f) for f in files]
    assert cli.main(["--manifest", str(manifest_path), "verify", "demo"]) == 0


def test_offline_without_cache_fails_clearly(tmp_path):
    _files, _out, entry = _pack(tmp_path)
    manifest = samples.Manifest(dict(samples.empty_manifest(), base_url="http://127.0.0.1:9/",
                                     stacks=[entry]))
    with pytest.raises(samples.OfflineError):
        samples.fetch_stack(manifest.get("demo"), manifest, offline=True)


def test_wrong_hash_leaves_no_valid_looking_stack(tmp_path, server):
    _files, _out, entry = _pack(tmp_path)
    entry["archives"][0]["sha256"] = "0" * 64
    manifest = _http_manifest(tmp_path, server, entry)
    with pytest.raises(samples.IntegrityError, match="SHA-256"):
        samples.fetch_stack(manifest.get("demo"), manifest)
    cache = tmp_path / "cache"
    assert samples.available_folder(manifest.get("demo"), manifest) is None
    assert not list(cache.rglob(samples.MARKER_NAME))
    assert not [p for p in (cache / "downloads").iterdir() if not p.name.endswith(".lock")]
    assert not [p for p in (cache / "stacks" / "demo").iterdir() if p.name != ".lock"]


def test_wrong_size_is_rejected(tmp_path, server):
    _files, _out, entry = _pack(tmp_path)
    entry["archives"][0]["bytes"] += 512
    manifest = _http_manifest(tmp_path, server, entry)
    with pytest.raises(samples.IntegrityError, match="bytes"):
        samples.fetch_stack(manifest.get("demo"), manifest)
    entry["archives"][0]["bytes"] -= 1024
    manifest = _http_manifest(tmp_path, server, entry)
    with pytest.raises(samples.IntegrityError, match="bytes"):
        samples.fetch_stack(manifest.get("demo"), manifest)
    assert not list((tmp_path / "cache").rglob(samples.MARKER_NAME))


def test_interrupted_download_resumes_on_retry(tmp_path, server):
    _files, _out, entry = _pack(tmp_path, count=4, size=200_000)
    manifest = _http_manifest(tmp_path, server, entry)
    stack = manifest.get("demo")
    server.truncate_next = 1
    with pytest.raises(samples.DownloadError, match="run again to resume"):
        samples.fetch_stack(stack, manifest, retries=0)
    part = tmp_path / "cache" / "downloads" / (entry["archives"][0]["sha256"] + ".part")
    kept = part.stat().st_size
    assert 0 < kept < entry["archives"][0]["bytes"]
    assert samples.available_folder(stack, manifest) is None
    assert not list((tmp_path / "cache").rglob(samples.MARKER_NAME))
    folder = samples.fetch_stack(stack, manifest, retries=0)
    assert server.requests[-1][1] == f"bytes={kept}-"
    assert samples.verify_stack(stack, manifest).ok
    assert not part.exists()
    assert folder.is_dir()


def test_interrupted_download_retries_automatically(tmp_path, server):
    _files, _out, entry = _pack(tmp_path, count=3, size=100_000)
    manifest = _http_manifest(tmp_path, server, entry)
    server.truncate_next = 2
    samples.fetch_stack(manifest.get("demo"), manifest, retries=3)
    assert [r[1] is not None for r in server.requests] == [False, True, True]
    # The cache is used without another request.
    samples.fetch_stack(manifest.get("demo"), manifest)
    assert len(server.requests) == 3


def test_multi_part_stack_and_subset_variant(tmp_path, server):
    files, out, entry = _pack(tmp_path, count=6, size=20_000, part_bytes=70_000)
    assert len(entry["archives"]) == 3
    assert all(a["bytes"] <= 70_000 for a in entry["archives"])
    entry["variants"] = [samples.subset_variant(entry, files[:2], "demo-first2")]
    manifest = _http_manifest(tmp_path, server, entry)
    paths = samples.stack_files("demo-first2", manifest)
    assert [p.name for p in paths] == [f.name for f in files[:2]]
    assert [r[0] for r in server.requests] == ["/demo.part01.tar"]
    paths = samples.stack_files("demo", manifest)
    assert [p.read_bytes() for p in paths] == [f.read_bytes() for f in files]
    assert sorted(r[0] for r in server.requests) == [
        "/demo.part01.tar", "/demo.part01.tar", "/demo.part02.tar", "/demo.part03.tar"]
    assert samples.available_folder(manifest.get("demo-first2"), manifest) is not None


def test_folder_without_marker_is_not_a_stack(tmp_path):
    files, out, entry = _pack(tmp_path)
    manifest = samples.Manifest(dict(samples.empty_manifest(), stacks=[entry]),
                                str(out / "m.json"))
    stack = manifest.get("demo")
    folder = tmp_path / "cache" / "stacks" / "demo" / stack.content_key
    folder.mkdir(parents=True)
    for path in files:
        (folder / path.name).write_bytes(path.read_bytes())
    assert samples.available_folder(stack, manifest) is None
    samples.write_json_atomic(folder / samples.MARKER_NAME, {"key": stack.content_key, "frames": []})
    assert samples.available_folder(stack, manifest) is None
    assert samples.fetch_stack(stack, manifest) == folder
    assert samples.available_folder(stack, manifest) == (folder, "cached")


# ---- unsafe archives ----------------------------------------------------------------

def _tar_with(path, build):
    with tarfile.open(path, "w") as bundle:
        build(bundle)


def _add(bundle, name, data=b"x", kind=tarfile.REGTYPE, link=""):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = link
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    bundle.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)


@pytest.mark.parametrize("member", [
    dict(name="../evil.jpg"), dict(name="/tmp/evil.jpg"), dict(name="a/../../evil.jpg"),
    dict(name="C:/evil.jpg"), dict(name="a\\..\\evil.jpg"),
    dict(name="link.jpg", kind=tarfile.SYMTYPE, link="/etc/passwd"),
    dict(name="hard.jpg", kind=tarfile.LNKTYPE, link="frame_1.jpg"),
    dict(name="fifo", kind=tarfile.FIFOTYPE),
])
def test_unsafe_tar_members_are_rejected(tmp_path, member):
    archive = tmp_path / "bad.tar"
    _tar_with(archive, lambda b: (_add(b, "frame_1.jpg"), _add(b, **member)))
    target = tmp_path / "out" / "target"
    target.mkdir(parents=True)
    with pytest.raises(samples.UnsafeArchiveError):
        samples.extract_members(archive, target, ["frame_1.jpg"])
    assert not any(target.iterdir())
    assert not (tmp_path / "out" / "evil.jpg").exists()


def test_unsafe_zip_members_are_rejected(tmp_path):
    for name, symlink in (("../evil.jpg", False), ("link.jpg", True)):
        archive = tmp_path / "bad.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("frame_1.jpg", b"x")
            info = zipfile.ZipInfo(name)
            if symlink:
                info.external_attr = 0o120777 << 16
            bundle.writestr(info, b"/etc/passwd")
        target = tmp_path / "zip-target"
        target.mkdir(exist_ok=True)
        with pytest.raises(samples.UnsafeArchiveError):
            samples.extract_members(archive, target, ["frame_1.jpg"])
        assert not any(target.iterdir())


def test_safe_zip_extracts_wanted_members_only(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("sub/", b"")
        bundle.writestr("sub/a.jpg", b"a")
        bundle.writestr("readme.txt", b"r")
    target = tmp_path / "t"
    target.mkdir()
    assert samples.extract_members(archive, target, ["sub/a.jpg"]) == ["sub/a.jpg"]
    assert (target / "sub" / "a.jpg").read_bytes() == b"a"
    assert not (target / "readme.txt").exists()
    with pytest.raises(samples.IntegrityError, match="lacks"):
        samples.extract_members(archive, tmp_path / "t2", ["b.jpg"])


# ---- local entries ----------------------------------------------------------------------

def test_local_entry_uses_the_folder_and_verify_detects_changes(tmp_path, capsys):
    files = _frames(tmp_path / "photos", 4)
    (tmp_path / "photos" / "notes.txt").write_text("not a frame")
    manifest_path = tmp_path / "local.json"
    assert cli.main(["pack", "-q", "--id", "local", "--local-only", "--folder",
                     str(tmp_path / "photos"), "--add-to", str(manifest_path),
                     "--no-exiftool", "--width", "8", "--height", "8", "--bit-depth", "8",
                     "--useful-for", "tests", "--tag", "synthetic"]) == 0
    assert cli.main(["pack", "-q", "--id", "local-first2", "--variant-of", "local",
                     "--add-to", str(manifest_path), *map(str, files[:2])]) == 0
    manifest = samples.load_manifest(manifest_path)
    assert samples.stack_files("local", manifest) == files
    assert samples.stack_files("local-first2", manifest) == files[:2]
    capsys.readouterr()
    assert cli.main(["--manifest", str(manifest_path), "path", "local-first2"]) == 1
    assert "--files" in capsys.readouterr().err
    assert cli.main(["--manifest", str(manifest_path), "path", "local"]) == 0
    assert cli.main(["--manifest", str(manifest_path), "list"]) == 0
    listing = capsys.readouterr().out
    assert "local-first2" in listing and "local" in listing
    files[1].write_bytes(files[1].read_bytes()[::-1])
    result = samples.verify_stack(manifest.get("local"), manifest)
    assert not result.ok and "frame_2.jpg: SHA-256" in result.problems[0]
    files[0].unlink()
    with pytest.raises(samples.SampleStackError, match="only available locally"):
        samples.stack_files("local", samples.load_manifest(manifest_path))


# ---- metadata ---------------------------------------------------------------------------

def _exif_jpeg(path, gps=True, serial=True, artist="Example Person"):
    """Minimal JPEG whose EXIF holds an artist, a body serial number and a GPS position."""
    order = "<"

    def ascii_entry(tag, text):
        data = text.encode() + b"\0"
        return (tag, 2, len(data), data)

    def ifd(entries, start):
        """Serialise an IFD at offset ``start``; returns bytes."""
        head = struct.pack(order + "H", len(entries))
        data_offset = start + 2 + 12 * len(entries) + 4
        table = b""
        extra = b""
        for tag, kind, count, data in entries:
            if len(data) <= 4:
                table += struct.pack(order + "HHI", tag, kind, count) + data.ljust(4, b"\0")
            else:
                table += struct.pack(order + "HHII", tag, kind, count, data_offset + len(extra))
                extra += data
        return head + table + struct.pack(order + "I", 0) + extra

    gps_entries = [(1, 2, 2, b"N\0"), (2, 5, 3, struct.pack(order + "6I", 47, 1, 36, 1, 0, 1))]
    exif_entries = [ascii_entry(0xA431, "1234567")] if serial else []
    zero = struct.pack(order + "I", 0)
    ifd0 = [ascii_entry(0x013B, artist)] if artist else []
    ifd0 += [(0x8769, 4, 1, zero)]
    if gps:
        ifd0 += [(0x8825, 4, 1, zero)]
    ifd0.sort()
    size0 = len(ifd(ifd0, 8))
    exif_start = 8 + size0
    exif_block = ifd(exif_entries, exif_start)
    gps_start = exif_start + len(exif_block)
    ifd0 = [(t, k, c, struct.pack(order + "I", exif_start) if t == 0x8769 else
             struct.pack(order + "I", gps_start) if t == 0x8825 else d) for t, k, c, d in ifd0]
    tiff = b"II*\0" + struct.pack(order + "I", 8) + ifd(ifd0, 8) + exif_block
    if gps:
        tiff += ifd(gps_entries, gps_start)
    app1 = b"Exif\0\0" + tiff
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, 4, 6, 1) + b"\x01\x11\x00"
    path.write_bytes(b"\xff\xd8\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1 + sof + b"\xff\xd9")
    return path


@pytest.mark.parametrize("use_exiftool", [False, True])
def test_metadata_check_finds_gps_serials_and_names(tmp_path, use_exiftool):
    if use_exiftool and samples.find_exiftool() is None:
        pytest.skip("exiftool is not installed")
    tagged = _exif_jpeg(tmp_path / "tagged.jpg")
    clean = _exif_jpeg(tmp_path / "clean.jpg", gps=False, serial=False, artist="")
    report = samples.inspect_metadata([tagged, clean], use_exiftool=use_exiftool)
    assert set(report.findings) >= {"gps", "serial_numbers", "names"}
    assert report.blocking == ["gps", "serial_numbers"]
    assert report.findings["names"]["values"] == {"Example Person"}
    assert all(len(v["files"]) == 1 for v in report.findings.values())
    text = "\n".join(report.summary_lines())
    assert "1234567" not in text and "47" not in text
    if not use_exiftool:
        assert any("maker notes" in item for item in report.not_checked)
    assert samples.image_shape(tagged, use_exiftool=use_exiftool)[:2] == (6, 4)


def test_pack_refuses_identifying_metadata_without_flag(tmp_path, capsys):
    folder = tmp_path / "photos"
    folder.mkdir()
    _exif_jpeg(folder / "a.jpg")
    _exif_jpeg(folder / "b.jpg", gps=False)
    arguments = ["pack", "--id", "tagged", "--folder", str(folder), "--out", str(tmp_path / "out"),
                 "--no-exiftool", "-q"]
    assert cli.main(arguments) == 3
    error = capsys.readouterr().err
    assert "GPS" in error and "serial numbers" in error and "--allow-identifying-metadata" in error
    assert not (tmp_path / "out").exists()
    assert cli.main(arguments + ["--allow-identifying-metadata"]) == 0
    assert (tmp_path / "out" / "tagged.tar").is_file()
    assert cli.main(["metadata", "--no-exiftool", str(folder)]) == 3
    assert cli.main(["metadata", "--no-exiftool", str(_frames(tmp_path / "plain", 1)[0])]) == 0


# ---- environment -----------------------------------------------------------------------------

def test_cache_location_overrides(tmp_path, monkeypatch):
    assert samples.cache_root(tmp_path / "x") == (tmp_path / "x").absolute()
    assert samples.cache_root() == (tmp_path / "cache").absolute()
    monkeypatch.delenv(samples.CACHE_ENV)
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "data"))
    assert samples.cache_root() == (tmp_path / "data" / "sample-stacks").absolute()


def test_library_and_cli_import_without_gui_or_numeric_packages(tmp_path):
    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('PyQt6', 'numpy', 'cv2', 'torch', 'PIL'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import utils.sample_stacks\n"
        "from tools import sample_stacks\n"
        "assert sample_stacks.main(['cache-dir']) == 0\n"
        "assert not [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'numpy', 'cv2')]\n"
    )
    environment = dict(os.environ, OPENFOCUS_SAMPLE_CACHE=str(tmp_path / "c"))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=60, env=environment)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "c")


def test_remote_manifest_is_cached_for_offline_use(tmp_path, server):
    _files, out, entry = _pack(tmp_path)
    _write_manifest(out / "manifest.json", [entry])
    url = server.url + "/manifest.json"
    manifest = samples.load_manifest(url)
    assert manifest.get("demo").archives[0].url == server.url + "/demo.tar"
    samples.fetch_stack(manifest.get("demo"), manifest)
    server.close()
    offline = samples.load_manifest(url, offline=True)
    assert len(samples.stack_files("demo", offline, offline=True)) == 5
    # A failing network falls back to the cached copy as well.
    assert samples.load_manifest(url).get("demo").title == "Demo"
