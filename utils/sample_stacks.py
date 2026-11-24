"""Sample focus stacks kept outside OpenFocus: manifest, download cache and packing.

The stacks themselves are published separately (for example as release assets
of a data-only repository). OpenFocus carries only this module and
``tools/sample_stacks.py``. See ``docs/SAMPLE_STACKS.md`` for the manifest
format and the publishing workflow.

Standard library only: no GUI, numerical or network packages are imported, so
benchmarks, tests and the headless wheel can use it.

Typical use from a benchmark or test::

    from utils.sample_stacks import stack_files
    paths = stack_files("my-stack-id")   # fetches and verifies on first use
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

SCHEMA = "openfocus-sample-stacks/1"
MANIFEST_ENV = "OPENFOCUS_SAMPLE_MANIFEST"
CACHE_ENV = "OPENFOCUS_SAMPLE_CACHE"
# Hosting limit for one release asset (2 GiB); every archive must be smaller.
MAX_ASSET_BYTES = 2 * 1024 ** 3
# Default archive part size: a comfortable margin below MAX_ASSET_BYTES.
DEFAULT_PART_BYTES = 1_900_000_000
MARKER_NAME = ".openfocus-sample-stack.json"
PLACEHOLDER_PREFIX = "TODO"
USER_AGENT = "OpenFocus-sample-stacks/1"
VARIANT_KINDS = ("subset", "crop", "downscale", "other")
IMAGE_EXTENSIONS = frozenset((
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif",
    ".jxl", ".dng", ".orf", ".ori", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw",
    ".srf", ".sr2", ".rw2", ".raf", ".pef", ".srw", ".x3f", ".erf", ".kdc", ".dcr",
    ".mrw", ".3fr", ".fff", ".iiq", ".rwl", ".mos", ".raw",
))
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPDX_RE = re.compile(
    r"^\(?[A-Za-z0-9.+-]+\)?(\s+(AND|OR|WITH)\s+\(?[A-Za-z0-9.+-]+\)?)*$")
_CHUNK = 1024 * 1024
_STACK_KEYS = {
    "id", "title", "description", "useful_for", "tags", "frame_count", "format",
    "width", "height", "bit_depth", "total_bytes", "license", "attribution",
    "frames", "archives", "local_path", "recommended", "variants", "references",
    "notes",
}
_VARIANT_KEYS = _STACK_KEYS - {"variants"} | {"kind"}
_RECOMMENDED_KEYS = {"alignment", "method", "parameters", "align_parameters",
                     "load_scale", "cli_args", "notes"}

ProgressCallback = Callable[[str, int, Optional[int]], None]


# ---- errors -----------------------------------------------------------------

class SampleStackError(Exception):
    """Base class for every sample-stack failure reported to a user."""


class ManifestNotConfigured(SampleStackError):
    pass


class ManifestError(SampleStackError):
    def __init__(self, message: str, issues: Sequence["Issue"] = ()):
        super().__init__(message)
        self.issues = list(issues)


class IntegrityError(SampleStackError):
    """Downloaded or local data does not match the manifest."""


class DownloadError(SampleStackError):
    """A download failed; a partial file may remain for resumption."""


class OfflineError(SampleStackError):
    pass


class UnsafeArchiveError(SampleStackError):
    pass


class IdentifyingMetadataError(SampleStackError):
    def __init__(self, message: str, report: "MetadataReport"):
        super().__init__(message)
        self.report = report


# ---- manifest model -----------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    level: str  # "error" or "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


@dataclass(frozen=True)
class Frame:
    path: str
    bytes: int
    sha256: str
    archive: Optional[int] = None
    url: Optional[str] = None


@dataclass(frozen=True)
class Archive:
    url: str
    bytes: int
    sha256: str
    name: str


@dataclass
class Stack:
    id: str
    title: str
    data: dict
    frames: List[Frame]
    archives: List[Archive]
    local_path: Optional[Path]
    parent: Optional[str] = None
    kind: Optional[str] = None
    variant_ids: List[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(frame.bytes for frame in self.frames)

    @property
    def content_key(self) -> str:
        """Short digest of the ordered frame list; a content change gets a new folder."""
        encoded = json.dumps([[f.path, f.bytes, f.sha256] for f in self.frames],
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _issue(issues, level, where, message):
    issues.append(Issue(level, where, message))


def safe_relative_path(name: str) -> Optional[PurePosixPath]:
    """Return a normalised relative path, or None when the name could escape a folder."""
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        return None
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return PurePosixPath(*parts)


def _is_int(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _check_strings(value, where, key, issues, required=True):
    if value is None:
        if required:
            _issue(issues, "error", where, f"'{key}' is required (a list of strings)")
        return
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        _issue(issues, "error", where, f"'{key}' must be a list of non-empty strings")


def _check_frames(entry, where, issues, archives_count, has_local):
    frames = entry.get("frames")
    if not isinstance(frames, list) or not frames:
        _issue(issues, "error", where, "'frames' must be a non-empty list (the frame order)")
        return []
    seen = set()
    total = 0
    used_archives = set()
    for index, frame in enumerate(frames):
        fwhere = f"{where}.frames[{index}]"
        if not isinstance(frame, dict):
            _issue(issues, "error", fwhere, "must be an object with path, bytes and sha256")
            continue
        unknown = set(frame) - {"path", "bytes", "sha256", "archive", "url"}
        if unknown:
            _issue(issues, "warning", fwhere, f"unknown keys {sorted(unknown)}")
        path = frame.get("path")
        if safe_relative_path(path) is None:
            _issue(issues, "error", fwhere, f"unsafe or missing relative path {path!r}")
        elif str(safe_relative_path(path)) != path:
            _issue(issues, "error", fwhere, f"path {path!r} must be written in normal form")
        elif path in seen:
            _issue(issues, "error", fwhere, f"duplicate frame path {path!r}")
        seen.add(path)
        if not _is_int(frame.get("bytes"), 1):
            _issue(issues, "error", fwhere, "'bytes' must be a positive integer")
        else:
            total += frame["bytes"]
        if not isinstance(frame.get("sha256"), str) or not _SHA256_RE.match(frame["sha256"]):
            _issue(issues, "error", fwhere, "'sha256' must be 64 lowercase hexadecimal digits")
        archive = frame.get("archive")
        if archive is not None:
            if not _is_int(archive) or archive >= archives_count:
                _issue(issues, "error", fwhere, f"'archive' {archive!r} is not an index into 'archives'")
            else:
                used_archives.add(archive)
        url = frame.get("url")
        if url is not None:
            _check_url(url, fwhere, issues)
        if archive is None and url is None and not has_local:
            _issue(issues, "error", fwhere,
                   "cannot be obtained: give an 'archive' index, a 'url' or the stack's 'local_path'")
    for index in range(archives_count):
        if index not in used_archives:
            _issue(issues, "warning", f"{where}.archives[{index}]", "no frame refers to this archive")
    if _is_int(entry.get("frame_count")) and entry["frame_count"] != len(frames):
        _issue(issues, "error", where,
               f"'frame_count' is {entry['frame_count']} but 'frames' lists {len(frames)}")
    if _is_int(entry.get("total_bytes")) and entry["total_bytes"] != total and total:
        _issue(issues, "error", where,
               f"'total_bytes' is {entry['total_bytes']} but the frames add up to {total}")
    return frames


def _check_url(url, where, issues):
    if not isinstance(url, str) or not url.strip():
        _issue(issues, "error", where, "'url' must be a non-empty string")
        return
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ("", "http", "https", "file"):
        _issue(issues, "error", where, f"unsupported URL scheme {scheme!r} (use https)")
    elif scheme == "http":
        _issue(issues, "warning", where, "plain http URL; prefer https")


def _check_archives(entry, where, issues):
    archives = entry.get("archives", [])
    if not isinstance(archives, list):
        _issue(issues, "error", where, "'archives' must be a list")
        return 0
    names = set()
    for index, archive in enumerate(archives):
        awhere = f"{where}.archives[{index}]"
        if not isinstance(archive, dict):
            _issue(issues, "error", awhere, "must be an object with url, bytes and sha256")
            continue
        _check_url(archive.get("url"), awhere, issues)
        size = archive.get("bytes")
        if not _is_int(size, 1):
            _issue(issues, "error", awhere, "'bytes' must be a positive integer")
        elif size >= MAX_ASSET_BYTES:
            _issue(issues, "error", awhere,
                   f"{size} bytes is not below the 2 GiB asset limit; split it into parts")
        if not isinstance(archive.get("sha256"), str) or not _SHA256_RE.match(archive["sha256"]):
            _issue(issues, "error", awhere, "'sha256' must be 64 lowercase hexadecimal digits")
        name = archive.get("name") or _url_name(archive.get("url") or "")
        if name in names:
            _issue(issues, "warning", awhere, f"archive name {name!r} repeats")
        names.add(name)
    return len(archives)


def _check_license(entry, where, issues):
    value = entry.get("license")
    if not isinstance(value, str) or not value.strip():
        _issue(issues, "error", where, "'license' is required (an SPDX identifier)")
    elif value.startswith(PLACEHOLDER_PREFIX):
        _issue(issues, "warning", where, f"'license' is a placeholder: {value}")
    elif not _SPDX_RE.match(value.strip()):
        _issue(issues, "error", where, f"'license' {value!r} is not an SPDX identifier or expression")
    attribution = entry.get("attribution")
    if not isinstance(attribution, str) or not attribution.strip():
        _issue(issues, "error", where, "'attribution' is required")
    elif attribution.startswith(PLACEHOLDER_PREFIX):
        _issue(issues, "warning", where, f"'attribution' is a placeholder: {attribution}")


def _check_recommended(value, where, issues):
    if value is None:
        return
    if not isinstance(value, dict):
        _issue(issues, "error", where, "'recommended' must be an object")
        return
    unknown = set(value) - _RECOMMENDED_KEYS
    if unknown:
        _issue(issues, "warning", where, f"unknown 'recommended' keys {sorted(unknown)}")
    for key in ("alignment", "method", "notes"):
        if key in value and not isinstance(value[key], str):
            _issue(issues, "error", where, f"'recommended.{key}' must be a string")
    for key in ("parameters", "align_parameters"):
        if key in value and not isinstance(value[key], dict):
            _issue(issues, "error", where, f"'recommended.{key}' must be an object")
    if "load_scale" in value:
        scale = value["load_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1:
            _issue(issues, "error", where, "'recommended.load_scale' must be a number in (0, 1]")
    if "cli_args" in value and (not isinstance(value["cli_args"], list)
                                or not all(isinstance(v, str) for v in value["cli_args"])):
        _issue(issues, "error", where, "'recommended.cli_args' must be a list of strings")


def _check_description_fields(entry, where, issues, required):
    for key in ("title", "description"):
        value = entry.get(key)
        if value is None and not required:
            continue
        if not isinstance(value, str) or not value.strip():
            _issue(issues, "error", where, f"'{key}' must be a non-empty string")
        elif value.startswith(PLACEHOLDER_PREFIX):
            _issue(issues, "warning", where, f"'{key}' is a placeholder: {value}")
    if required or "format" in entry:
        if not isinstance(entry.get("format"), str) or not entry.get("format"):
            _issue(issues, "error", where, "'format' must be a string such as ORF, JPEG or TIFF")
    for key in ("width", "height", "bit_depth"):
        if required or key in entry:
            if not _is_int(entry.get(key), 1):
                _issue(issues, "error", where, f"'{key}' must be a positive integer")
    for key in ("frame_count", "total_bytes"):
        if (required or key in entry) and not _is_int(entry.get(key), 1):
            _issue(issues, "error", where, f"'{key}' must be a positive integer")
    _check_strings(entry.get("useful_for"), where, "useful_for", issues, required)
    _check_strings(entry.get("tags"), where, "tags", issues, required)
    if "references" in entry and not isinstance(entry["references"], list):
        _issue(issues, "error", where, "'references' is reserved and must be a list when present")
    if "local_path" in entry and (not isinstance(entry["local_path"], str) or not entry["local_path"]):
        _issue(issues, "error", where, "'local_path' must be a folder path string")


def validate_manifest(data) -> List[Issue]:
    """Return every problem found in manifest data (errors and warnings)."""
    issues: List[Issue] = []
    if not isinstance(data, dict):
        return [Issue("error", "manifest", "must be a JSON object")]
    if data.get("schema") != SCHEMA:
        _issue(issues, "error", "manifest",
               f"'schema' must be {SCHEMA!r} (found {data.get('schema')!r})")
    for key in set(data) - {"schema", "title", "description", "base_url", "stacks"}:
        if not key.startswith("x-"):
            _issue(issues, "warning", "manifest", f"unknown key {key!r}")
    if "base_url" in data:
        _check_url(data["base_url"], "manifest.base_url", issues)
    stacks = data.get("stacks")
    if not isinstance(stacks, list):
        _issue(issues, "error", "manifest", "'stacks' must be a list")
        return issues
    ids = set()

    def claim(identifier, where):
        if not isinstance(identifier, str) or not _ID_RE.match(identifier):
            _issue(issues, "error", where,
                   f"'id' {identifier!r} must be 1-80 of a-z, 0-9, '.', '_' or '-', "
                   "starting with a letter or digit")
        elif identifier in ids:
            _issue(issues, "error", where, f"duplicate id {identifier!r}")
        ids.add(identifier)

    for index, entry in enumerate(stacks):
        where = f"stacks[{index}]"
        if not isinstance(entry, dict):
            _issue(issues, "error", where, "must be an object")
            continue
        where = f"stacks[{index}] ({entry.get('id')})"
        claim(entry.get("id"), where)
        for key in set(entry) - _STACK_KEYS:
            if not key.startswith("x-"):
                _issue(issues, "warning", where, f"unknown key {key!r}")
        _check_description_fields(entry, where, issues, required=True)
        _check_license(entry, where, issues)
        _check_recommended(entry.get("recommended"), where, issues)
        archives = _check_archives(entry, where, issues)
        frames = _check_frames(entry, where, issues, archives, "local_path" in entry)
        parent_paths = {f.get("path") for f in frames if isinstance(f, dict)}
        variants = entry.get("variants", [])
        if not isinstance(variants, list):
            _issue(issues, "error", where, "'variants' must be a list")
            continue
        for vindex, variant in enumerate(variants):
            vwhere = f"{where}.variants[{vindex}]"
            if not isinstance(variant, dict):
                _issue(issues, "error", vwhere, "must be an object")
                continue
            vwhere = f"{vwhere} ({variant.get('id')})"
            claim(variant.get("id"), vwhere)
            for key in set(variant) - _VARIANT_KEYS:
                if not key.startswith("x-"):
                    _issue(issues, "warning", vwhere, f"unknown key {key!r}")
            kind = variant.get("kind")
            if kind not in VARIANT_KINDS:
                _issue(issues, "error", vwhere, f"'kind' must be one of {list(VARIANT_KINDS)}")
            _check_description_fields(variant, vwhere, issues, required=False)
            for key in ("title", "description"):
                if key not in variant:
                    _issue(issues, "error", vwhere, f"'{key}' is required")
            _check_recommended(variant.get("recommended"), vwhere, issues)
            if "license" in variant or "attribution" in variant:
                merged = {"license": entry.get("license"), "attribution": entry.get("attribution")}
                merged.update({k: variant[k] for k in ("license", "attribution") if k in variant})
                _check_license(merged, vwhere, issues)
            if kind == "subset":
                names = variant.get("frames")
                if (not isinstance(names, list) or not names
                        or not all(isinstance(n, str) for n in names)):
                    _issue(issues, "error", vwhere,
                           "a subset variant's 'frames' must list parent frame paths")
                    continue
                missing = [n for n in names if n not in parent_paths]
                if missing:
                    _issue(issues, "error", vwhere, f"frames not in the parent stack: {missing[:5]}")
                if len(set(names)) != len(names):
                    _issue(issues, "error", vwhere, "a frame is listed twice")
                for key in ("archives", "local_path"):
                    if key in variant:
                        _issue(issues, "error", vwhere,
                               f"a subset variant takes '{key}' from its parent")
                if _is_int(variant.get("frame_count")) and variant["frame_count"] != len(names):
                    _issue(issues, "error", vwhere,
                           f"'frame_count' is {variant['frame_count']} but 'frames' lists {len(names)}")
            elif kind in VARIANT_KINDS:
                vcount = _check_archives(variant, vwhere, issues)
                _check_frames(variant, vwhere, issues, vcount, "local_path" in variant)
                for key in ("format", "width", "height", "bit_depth", "frame_count", "total_bytes"):
                    if key not in variant:
                        _issue(issues, "error", vwhere, f"'{key}' is required for a {kind} variant")
    return issues


def _url_name(url: str) -> str:
    return PurePosixPath(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name or "download"


class Manifest:
    """A validated manifest with variants resolved into addressable stacks."""

    def __init__(self, data: dict, location: Optional[str] = None,
                 issues: Optional[List[Issue]] = None):
        self.data = data
        self.location = location
        self.issues = issues if issues is not None else validate_manifest(data)
        errors = [i for i in self.issues if i.level == "error"]
        if errors:
            where = f" {location}" if location else ""
            raise ManifestError(f"manifest{where} has {len(errors)} error(s):\n"
                                + "\n".join(f"  {e}" for e in errors), self.issues)
        self.stacks: Dict[str, Stack] = {}
        for entry in data["stacks"]:
            parent = self._build(entry, None)
            for variant in entry.get("variants", []):
                built = self._build(variant, parent)
                parent.variant_ids.append(built.id)

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def _local_base(self) -> Optional[Path]:
        if self.location and not _is_remote(self.location):
            return Path(self.location).expanduser().absolute().parent
        return None

    def resolve_url(self, reference: str) -> str:
        """Resolve a relative URL against base_url, else against the manifest's location."""
        if urllib.parse.urlsplit(reference).scheme:
            return reference
        base = self.data.get("base_url")
        if not base:
            if self.location and _is_remote(self.location):
                base = self.location
            elif self._local_base() is not None:
                base = self._local_base().as_uri() + "/"
            else:
                raise SampleStackError(f"relative URL {reference!r} needs 'base_url'")
        elif not base.endswith("/"):
            base += "/"
        return urllib.parse.urljoin(base, reference)

    def _resolve_local(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            base = self._local_base()
            path = (base / path) if base is not None else path.absolute()
        return path

    def _build(self, entry: dict, parent: Optional[Stack]) -> Stack:
        if parent is not None and entry.get("kind") == "subset":
            by_path = {frame.path: frame for frame in parent.frames}
            data = {k: v for k, v in parent.data.items()
                    if k not in ("id", "title", "description", "variants", "frames",
                                 "useful_for", "tags", "recommended", "references")}
            data.update(entry)
            frames = [by_path[name] for name in entry["frames"]]
            data["frame_count"] = len(frames)
            data["total_bytes"] = sum(f.bytes for f in frames)
            data.setdefault("useful_for", parent.data.get("useful_for", []))
            data.setdefault("tags", parent.data.get("tags", []))
            stack = Stack(entry["id"], entry["title"], data, frames, parent.archives,
                          parent.local_path, parent=parent.id, kind="subset")
        else:
            data = dict(entry)
            if parent is not None:
                for key in ("license", "attribution", "useful_for", "tags"):
                    data.setdefault(key, parent.data.get(key))
            archives = [Archive(self.resolve_url(a["url"]), a["bytes"], a["sha256"],
                                a.get("name") or _url_name(a["url"]))
                        for a in entry.get("archives", [])]
            frames = [Frame(f["path"], f["bytes"], f["sha256"], f.get("archive"),
                            self.resolve_url(f["url"]) if f.get("url") else None)
                      for f in entry["frames"]]
            stack = Stack(entry["id"], entry["title"], data, frames, archives,
                          self._resolve_local(entry.get("local_path")),
                          parent=parent.id if parent else None,
                          kind=entry.get("kind") if parent else None)
        self.stacks[stack.id] = stack
        return stack

    def get(self, stack_id: str) -> Stack:
        try:
            return self.stacks[stack_id]
        except KeyError:
            known = ", ".join(sorted(self.stacks)) or "(none)"
            raise SampleStackError(f"unknown stack {stack_id!r}; known stacks: {known}") from None


def _is_remote(location: str) -> bool:
    return urllib.parse.urlsplit(location).scheme.lower() in ("http", "https")


def manifest_source(source: Optional[Union[str, Path]] = None) -> str:
    """The manifest path or URL to use: the argument, else ``$OPENFOCUS_SAMPLE_MANIFEST``."""
    value = str(source) if source else os.environ.get(MANIFEST_ENV, "").strip()
    if not value:
        raise ManifestNotConfigured(
            "No sample-stack manifest is configured. Pass --manifest PATH_OR_URL or set "
            f"{MANIFEST_ENV}. OpenFocus does not ship one; see docs/SAMPLE_STACKS.md for "
            "the format and for writing a local manifest with 'pack --local-only'.")
    return value


def load_manifest(source: Optional[Union[str, Path]] = None, *,
                  cache_dir: Optional[Union[str, Path]] = None,
                  offline: bool = False) -> Manifest:
    """Load and validate a manifest from a path, a file:// URL or an https URL.

    A remote manifest is cached; offline (or when the network fails) the cached
    copy is used.
    """
    location = manifest_source(source)
    scheme = urllib.parse.urlsplit(location).scheme.lower()
    if scheme == "file":
        location = urllib.request.url2pathname(urllib.parse.urlsplit(location).path)
        scheme = ""
    if scheme in ("http", "https"):
        cached = cache_root(cache_dir) / "manifests" / (
            hashlib.sha256(location.encode("utf-8")).hexdigest()[:24] + ".json")
        raw = None
        if not offline:
            try:
                request = urllib.request.Request(location, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise SampleStackError(f"manifest {location} is larger than 16 MiB")
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                if not cached.is_file():
                    raise DownloadError(f"could not download manifest {location}: {exc}") from exc
                raw = None
        if raw is None:
            if not cached.is_file():
                raise OfflineError(f"manifest {location} is not cached; it is needed once online")
            raw = cached.read_bytes()
        data = _parse_json(raw, location)
        manifest = Manifest(data, location)
        if not offline:
            _atomic_write_bytes(cached, raw)
        return manifest
    path = Path(location).expanduser()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ManifestNotConfigured(f"manifest file {path} does not exist") from None
    return Manifest(_parse_json(raw, str(path)), str(path))


def _parse_json(raw: bytes, location: str):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest {location} is not valid UTF-8 JSON: {exc}") from exc


# ---- cache location and file helpers ------------------------------------------

def cache_root(override: Optional[Union[str, Path]] = None) -> Path:
    """Folder holding downloaded sample stacks.

    Order: the argument, ``$OPENFOCUS_SAMPLE_CACHE``, the OpenFocus data folder
    when it is isolated (``$OPENFOCUS_DATA_DIR``) or portable, else the
    platform's per-user cache folder.
    """
    if override:
        return Path(override).expanduser().absolute()
    value = os.environ.get(CACHE_ENV, "").strip()
    if value:
        return Path(value).expanduser().absolute()
    from utils import app_paths
    if os.environ.get("OPENFOCUS_DATA_DIR") or app_paths.is_portable_mode():
        return app_paths.user_data_dir() / "sample-stacks"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "OpenFocus" / "Cache" / "sample-stacks"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "OpenFocus" / "sample-stacks"
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "OpenFocus" / "sample-stacks"


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _fsync_dir(path.parent)


def write_json_atomic(path: Union[str, Path], data) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


class _FileLock:
    """Exclusive inter-process lock on a file (released when the process ends)."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
        else:
            import fcntl
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


# ---- downloads ------------------------------------------------------------------

class _Restart(Exception):
    """The partial file cannot be continued; start the download again."""


class _Incomplete(Exception):
    pass


_RETRYABLE = (urllib.error.URLError, http.client.HTTPException, ConnectionError,
              socket.timeout, TimeoutError, _Incomplete)


def download_verified(url: str, size: int, sha256: str, directory: Union[str, Path], *,
                      offline: bool = False, retries: int = 3, timeout: float = 60.0,
                      progress: Optional[ProgressCallback] = None,
                      label: Optional[str] = None, retry_delay: float = 1.0) -> Path:
    """Download ``url`` into ``directory/<sha256>`` and return it once verified.

    The data is written to ``<sha256>.part`` and only renamed into place after
    the size and SHA-256 match. An interrupted download keeps the partial file
    and the next call resumes it with an HTTP range request (or restarts when
    the server cannot resume). A wrong size or digest removes the partial file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / sha256
    part = directory / f"{sha256}.part"
    label = label or _url_name(url)
    with _FileLock(directory / f"{sha256}.lock"):
        if final.is_file() and final.stat().st_size == size:
            return final
        if offline:
            raise OfflineError(f"{label} is not in the cache and --offline was given")
        failures = 0
        restarted = False
        while True:
            resumed = part.is_file() and part.stat().st_size > 0
            try:
                have = part.stat().st_size if part.is_file() else 0
                if have > size:
                    part.unlink()
                    have = 0
                if have < size:
                    _transfer(url, part, have, size, timeout, progress, label)
                have = part.stat().st_size if part.is_file() else 0
                if have != size:
                    raise _Incomplete(f"received {have} of {size} bytes")
                digest = sha256_file(part)
                if digest != sha256:
                    part.unlink()
                    if resumed and not restarted:
                        # The resumed prefix may have come from a different file.
                        restarted = True
                        continue
                    raise IntegrityError(
                        f"{label}: SHA-256 is {digest}, the manifest expects {sha256}")
                os.replace(part, final)
                _fsync_dir(directory)
                return final
            except _Restart:
                if part.exists():
                    part.unlink()
                if restarted:
                    raise DownloadError(f"{label}: the server could not resume or restart cleanly")
                restarted = True
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise DownloadError(f"{label}: HTTP {exc.code} from {url}") from exc
                failures += 1
                if failures > retries:
                    raise DownloadError(f"{label}: HTTP {exc.code} from {url}; "
                                        f"partial data kept in {part}, run again to resume") from exc
                time.sleep(retry_delay * failures)
            except _RETRYABLE as exc:
                failures += 1
                if failures > retries:
                    kept = f"; partial data kept in {part}, run again to resume" if part.exists() else ""
                    raise DownloadError(f"{label}: download interrupted ({exc}){kept}") from exc
                time.sleep(retry_delay * failures)


def _transfer(url, part: Path, have: int, size: int, timeout, progress, label):
    split = urllib.parse.urlsplit(url)
    if split.scheme.lower() == "file":
        source = Path(urllib.request.url2pathname(split.path))
        try:
            actual = source.stat().st_size
        except FileNotFoundError:
            raise DownloadError(f"{label}: {source} does not exist") from None
        if actual != size:
            raise IntegrityError(f"{label}: {source} has {actual} bytes, the manifest says {size}")
        with open(source, "rb") as reader, open(part, "r+b" if have else "wb") as writer:
            reader.seek(have)
            writer.seek(have)
            _copy_stream(reader, writer, have, size, progress, label)
        return
    if split.scheme.lower() not in ("http", "https"):
        raise DownloadError(f"{label}: unsupported URL {url!r}")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:
            raise _Restart() from exc
        raise
    with response:
        status = response.status
        if have and status == 206:
            match = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
            if not match or int(match.group(1)) != have:
                raise _Restart()
            if match.group(3) != "*" and int(match.group(3)) != size:
                part.unlink()
                raise IntegrityError(f"{label}: the server reports {match.group(3)} bytes, "
                                     f"the manifest says {size}")
            mode = "ab"
        elif status == 200:
            have = 0
            mode = "wb"
            length = response.headers.get("Content-Length")
            if length is not None and length.isdigit() and int(length) != size:
                if part.exists():
                    part.unlink()
                raise IntegrityError(f"{label}: the server reports {length} bytes, "
                                     f"the manifest says {size}")
        else:
            raise DownloadError(f"{label}: unexpected HTTP status {status}")
        with open(part, mode) as writer:
            _copy_stream(response, writer, have, size, progress, label)


def _copy_stream(reader, writer, have, size, progress, label):
    done = have
    last = 0.0
    while True:
        block = reader.read(_CHUNK)
        if not block:
            break
        done += len(block)
        if done > size:
            writer.close()
            Path(writer.name).unlink()
            raise IntegrityError(f"{label}: more data than the {size} bytes the manifest lists")
        writer.write(block)
        now = time.monotonic()
        if progress and (now - last > 0.5 or done == size):
            last = now
            progress(label, done, size)
    writer.flush()
    os.fsync(writer.fileno())


# ---- archives ----------------------------------------------------------------------

def _zip_is_link(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def extract_members(archive: Union[str, Path], target: Union[str, Path],
                    wanted: Union[Iterable[str], Dict[str, int]]) -> List[str]:
    """Extract the regular-file members named in ``wanted`` into ``target``.

    ``wanted`` is a collection of member names, or a mapping from member name
    to its expected size in bytes. With sizes, a member whose archive header
    declares another size is rejected before anything is written, and copying
    stops with an error (removing the partial file) as soon as more bytes
    arrive than expected; fewer bytes are an error too. ``fetch_stack`` always
    passes the frame sizes from the manifest.

    Every member of the archive is checked first. Absolute names, ``..``,
    drive letters, backslashes, links and device entries reject the whole
    archive. Members that are safe but not wanted are skipped. Raises when a
    wanted member is missing. Returns the extracted names.
    """
    target = Path(target).absolute()
    sizes: Dict[str, Optional[int]] = (
        dict(wanted) if isinstance(wanted, dict) else dict.fromkeys(wanted))
    archive = Path(archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = []
            for info in bundle.infolist():
                name = info.filename
                relative = safe_relative_path(name.rstrip("/") if info.is_dir() else name)
                if relative is None:
                    raise UnsafeArchiveError(f"{archive.name}: unsafe member name {name!r}")
                if _zip_is_link(info):
                    raise UnsafeArchiveError(f"{archive.name}: link member {name!r}")
                if not info.is_dir():
                    members.append((str(relative), info))
            try:
                return _extract_all(archive, target, sizes, members,
                                    lambda info: bundle.open(info),
                                    lambda info: info.file_size)
            except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
                raise IntegrityError(f"{archive.name}: damaged zip member ({exc})") from exc
    try:
        bundle = tarfile.open(archive, "r:*")
    except tarfile.TarError as exc:
        raise IntegrityError(f"{archive.name} is neither a tar nor a zip archive: {exc}") from exc
    with bundle:
        members = []
        for info in bundle.getmembers():
            relative = safe_relative_path(info.name)
            if relative is None:
                raise UnsafeArchiveError(f"{archive.name}: unsafe member name {info.name!r}")
            if info.isdir():
                continue
            if not info.isreg():
                raise UnsafeArchiveError(
                    f"{archive.name}: member {info.name!r} is a link or special file")
            members.append((str(relative), info))
        try:
            return _extract_all(archive, target, sizes, members, bundle.extractfile,
                                lambda info: info.size)
        except (tarfile.TarError, zlib.error, EOFError) as exc:
            raise IntegrityError(f"{archive.name}: damaged tar member ({exc})") from exc


def _extract_all(archive, target: Path, sizes: Dict[str, Optional[int]], members, opener,
                 declared_size):
    wanted = set(sizes)
    names = [name for name, _ in members]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates & wanted:
        raise UnsafeArchiveError(f"{archive.name}: repeated members {sorted(duplicates & wanted)}")
    missing = wanted - set(names)
    if missing:
        raise IntegrityError(f"{archive.name} lacks {len(missing)} frame(s): {sorted(missing)[:5]}")
    for name, info in members:
        expected = sizes.get(name)
        if name in wanted and expected is not None and declared_size(info) != expected:
            raise IntegrityError(f"{archive.name}: {name} is {declared_size(info)} bytes in the "
                                 f"archive, the manifest says {expected}")
    extracted = []
    for name, info in members:
        if name not in wanted:
            continue
        destination = target.joinpath(*PurePosixPath(name).parts)
        if not str(destination.absolute()).startswith(str(target) + os.sep):
            raise UnsafeArchiveError(f"{archive.name}: {name!r} resolves outside the target")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.resolve() != target.resolve().joinpath(*PurePosixPath(name).parts[:-1]):
            raise UnsafeArchiveError(f"{archive.name}: {name!r} resolves outside the target")
        limit = sizes[name] if sizes[name] is not None else declared_size(info)
        writer = open(destination, "xb")  # never replaces an existing file
        try:
            with opener(info) as reader, writer:
                written = 0
                while True:
                    block = reader.read(min(_CHUNK, limit - written + 1))
                    if not block:
                        break
                    written += len(block)
                    if written > limit:
                        raise IntegrityError(f"{archive.name}: {name} holds more than the "
                                             f"{limit} bytes expected")
                    writer.write(block)
                if written != limit:
                    raise IntegrityError(f"{archive.name}: {name} holds {written} bytes, "
                                         f"expected {limit}")
                writer.flush()
                os.fsync(writer.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                destination.unlink()
            raise
        extracted.append(name)
    return extracted


# ---- fetching -------------------------------------------------------------------

def _stack_dir(root: Path, stack: Stack) -> Path:
    return root / "stacks" / stack.id / stack.content_key


def _read_marker(folder: Path) -> Optional[dict]:
    try:
        return json.loads((folder / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _marker_matches(folder: Path, stack: Stack) -> bool:
    marker = _read_marker(folder)
    if not marker or marker.get("key") != stack.content_key:
        return False
    if marker.get("frames") != [[f.path, f.bytes, f.sha256] for f in stack.frames]:
        return False
    return all(_size_ok(folder, frame) for frame in stack.frames)


def _size_ok(folder: Path, frame: Frame) -> bool:
    try:
        return (folder / frame.path).stat().st_size == frame.bytes
    except OSError:
        return False


def _frame_files(folder: Path, stack: Stack) -> List[Path]:
    return [folder.joinpath(*PurePosixPath(frame.path).parts) for frame in stack.frames]


def available_folder(stack: Stack, manifest: Optional[Manifest] = None,
                     cache_dir: Optional[Union[str, Path]] = None) -> Optional[Tuple[Path, str]]:
    """Where the stack's frames are now, without fetching: (folder, "local"|"cached")."""
    if stack.local_path is not None and stack.local_path.is_dir() and all(
            _size_ok(stack.local_path, frame) for frame in stack.frames):
        return stack.local_path, "local"
    root = cache_root(cache_dir)
    folder = _stack_dir(root, stack)
    if _marker_matches(folder, stack):
        return folder, "cached"
    if stack.parent and manifest is not None and stack.kind == "subset":
        parent = manifest.get(stack.parent)
        found = available_folder(parent, manifest, cache_dir)
        if found and found[1] == "cached":
            return found[0], "cached"
    return None


def fetch_stack(stack: Stack, manifest: Optional[Manifest] = None, *,
                cache_dir: Optional[Union[str, Path]] = None, offline: bool = False,
                keep_downloads: bool = False, retries: int = 3, timeout: float = 60.0,
                progress: Optional[ProgressCallback] = None) -> Path:
    """Make the stack available locally and return the folder holding its frames.

    A local_path that holds the frames (by size) is used directly. Otherwise the
    stack is assembled in a temporary folder, every frame's SHA-256 is checked,
    a completion marker is written, and only then is the folder renamed into
    place; a folder without a valid marker is never treated as a stack.
    """
    found = available_folder(stack, manifest, cache_dir)
    if found:
        return found[0]
    root = cache_root(cache_dir)
    folder = _stack_dir(root, stack)
    with _FileLock(root / "stacks" / stack.id / ".lock"):
        if _marker_matches(folder, stack):
            return folder
        for stale in folder.parent.glob(".tmp-*"):
            shutil.rmtree(stale, ignore_errors=True)
        if folder.exists():
            shutil.rmtree(folder)
        if offline:
            raise OfflineError(f"stack {stack.id!r} is not in the cache ({folder}) and "
                               "--offline was given")
        for frame in stack.frames:
            if frame.archive is None and frame.url is None:
                where = stack.local_path or "(no local_path)"
                raise SampleStackError(
                    f"stack {stack.id!r}: frame {frame.path} is only available locally, "
                    f"and {where} does not hold it with the listed size")
        temporary = folder.parent / f".tmp-{stack.content_key}-{os.getpid()}"
        temporary.mkdir(parents=True)
        downloads = root / "downloads"
        fetched: List[Path] = []
        try:
            by_archive: Dict[int, List[Frame]] = {}
            for frame in stack.frames:
                if frame.archive is not None:
                    by_archive.setdefault(frame.archive, []).append(frame)
            for index in sorted(by_archive):
                archive = stack.archives[index]
                path = download_verified(archive.url, archive.bytes, archive.sha256, downloads,
                                         retries=retries, timeout=timeout, progress=progress,
                                         label=archive.name)
                fetched.append(path)
                extract_members(path, temporary, {f.path: f.bytes for f in by_archive[index]})
            for frame in stack.frames:
                if frame.archive is None:
                    path = download_verified(frame.url, frame.bytes, frame.sha256, downloads,
                                             retries=retries, timeout=timeout, progress=progress,
                                             label=frame.path)
                    destination = temporary.joinpath(*PurePosixPath(frame.path).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
                    fetched.append(path)
            problems = _check_frames_on_disk(temporary, stack.frames)
            if problems:
                raise IntegrityError(f"stack {stack.id!r} failed verification after "
                                     "extraction:\n  " + "\n  ".join(problems))
            marker = {"schema": SCHEMA, "id": stack.id, "key": stack.content_key,
                      "frames": [[f.path, f.bytes, f.sha256] for f in stack.frames],
                      "completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            write_json_atomic(temporary / MARKER_NAME, marker)
            os.replace(temporary, folder)
            _fsync_dir(folder.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        if not keep_downloads:
            for path in fetched:
                with contextlib.suppress(OSError):
                    path.unlink()
        return folder


def _check_frames_on_disk(folder: Path, frames: Sequence[Frame]) -> List[str]:
    problems = []
    for frame in frames:
        path = folder.joinpath(*PurePosixPath(frame.path).parts)
        if not path.is_file():
            problems.append(f"{frame.path}: missing")
            continue
        size = path.stat().st_size
        if size != frame.bytes:
            problems.append(f"{frame.path}: {size} bytes, expected {frame.bytes}")
            continue
        digest = sha256_file(path)
        if digest != frame.sha256:
            problems.append(f"{frame.path}: SHA-256 {digest}, expected {frame.sha256}")
    return problems


@dataclass
class VerifyResult:
    stack_id: str
    folder: Optional[Path]
    source: Optional[str]
    problems: List[str]
    files: int
    bytes: int

    @property
    def ok(self) -> bool:
        return self.folder is not None and not self.problems


def verify_stack(stack: Stack, manifest: Optional[Manifest] = None, *,
                 cache_dir: Optional[Union[str, Path]] = None) -> VerifyResult:
    """Hash every frame of a local or cached stack against the manifest."""
    found = available_folder(stack, manifest, cache_dir)
    if not found:
        problems = [f"not available locally; run fetch {stack.id}"]
        if stack.local_path is not None:
            missing = [f.path for f in stack.frames if not _size_ok(stack.local_path, f)]
            problems = [f"local_path {stack.local_path}: {len(missing)} frame(s) missing "
                        f"or with another size, e.g. {missing[:3]}"]
        return VerifyResult(stack.id, None, None, problems, 0, 0)
    folder, source = found
    problems = _check_frames_on_disk(folder, stack.frames)
    return VerifyResult(stack.id, folder, source, problems, len(stack.frames), stack.total_bytes)


def _manifest_for(manifest, cache_dir, offline) -> Manifest:
    if isinstance(manifest, Manifest):
        return manifest
    return load_manifest(manifest, cache_dir=cache_dir, offline=offline)


def stack_files(stack_id: str, manifest: Union[Manifest, str, Path, None] = None, *,
                cache_dir: Optional[Union[str, Path]] = None, fetch: bool = True,
                offline: bool = False, progress: Optional[ProgressCallback] = None) -> List[Path]:
    """Ordered frame paths of a stack, fetching and verifying it first if needed.

    ``manifest`` is a loaded Manifest, a path or URL, or None for
    ``$OPENFOCUS_SAMPLE_MANIFEST``.
    """
    loaded = _manifest_for(manifest, cache_dir, offline)
    stack = loaded.get(stack_id)
    found = available_folder(stack, loaded, cache_dir)
    if found:
        return _frame_files(found[0], stack)
    if not fetch:
        raise SampleStackError(f"stack {stack_id!r} is not available locally")
    folder = fetch_stack(stack, loaded, cache_dir=cache_dir, offline=offline, progress=progress)
    return _frame_files(folder, stack)


def stack_folder(stack_id: str, manifest: Union[Manifest, str, Path, None] = None,
                 **kwargs) -> Path:
    """Folder holding a stack's frames (it may also hold other files; see path --files)."""
    return stack_files(stack_id, manifest, **kwargs)[0].parent


def folder_is_exact(folder: Path, stack: Stack) -> Tuple[bool, str]:
    """Whether loading ``folder`` as a stack yields exactly this stack's frames in order."""
    from utils.sorting import natural_sort_key
    if any("/" in frame.path for frame in stack.frames):
        return False, "frames are in subfolders"
    images = [p.name for p in folder.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    names = [frame.path for frame in stack.frames]
    extra = sorted(set(images) - set(names))
    if extra:
        return False, f"the folder holds {len(extra)} other image file(s)"
    if sorted(names, key=natural_sort_key) != names:
        return False, "the frame order differs from the folder's name order"
    return True, ""


# ---- metadata inspection --------------------------------------------------------------

BLOCKING_KINDS = ("gps", "serial_numbers")
KIND_LABELS = {
    "gps": "GPS position or other GPS data",
    "serial_numbers": "camera, lens or accessory serial numbers",
    "names": "owner, artist, author or copyright names",
    "unique_ids": "unique image or document identifiers",
}
_NAME_TAGS = {"artist", "copyright", "ownername", "cameraownername", "author", "creator",
              "by-line", "rights", "copyrightnotice", "xpauthor", "writer-editor", "credit",
              "owner", "copyrightowner", "creatorcontactinfo", "usercomment-owner"}
_ID_TAGS = {"imageuniqueid", "documentid", "originaldocumentid", "instanceid"}


@dataclass
class MetadataReport:
    tool: str
    files: int
    findings: Dict[str, Dict[str, set]] = field(default_factory=dict)
    not_checked: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add(self, kind: str, tag: str, path: str, value=None):
        entry = self.findings.setdefault(kind, {"tags": set(), "files": set(), "values": set()})
        entry["tags"].add(tag)
        entry["files"].add(path)
        if kind == "names" and value is not None:
            entry["values"].add(str(value).strip())

    @property
    def blocking(self) -> List[str]:
        return [kind for kind in BLOCKING_KINDS if kind in self.findings]

    def as_dict(self, show_values: bool = True) -> dict:
        return {
            "tool": self.tool, "files": self.files, "not_checked": self.not_checked,
            "errors": self.errors, "blocking": self.blocking,
            "findings": {kind: {"label": KIND_LABELS[kind], "files": len(v["files"]),
                                "tags": sorted(v["tags"]),
                                **({"values": sorted(v["values"])}
                                   if show_values and kind == "names" else {})}
                         for kind, v in sorted(self.findings.items())},
        }

    def summary_lines(self, show_values: bool = True) -> List[str]:
        lines = [f"Metadata check of {self.files} file(s) with {self.tool}:"]
        if not self.findings:
            lines.append("  no GPS data, serial numbers, names or unique identifiers found")
        for kind in ("gps", "serial_numbers", "names", "unique_ids"):
            if kind not in self.findings:
                continue
            entry = self.findings[kind]
            line = (f"  {KIND_LABELS[kind]}: {len(entry['files'])} file(s); "
                    f"tags {', '.join(sorted(entry['tags']))}")
            if kind == "names" and show_values and entry["values"]:
                line += f"; values {', '.join(sorted(entry['values']))}"
            lines.append(line)
        for item in self.not_checked:
            lines.append(f"  not checked: {item}")
        for item in self.errors:
            lines.append(f"  error: {item}")
        return lines


def _meaningful(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (list, tuple)):
        return any(_meaningful(v) for v in value)
    text = str(value).strip().strip("\x00").strip()
    if not text or text.lower() in ("none", "unknown", "n/a", "(none)"):
        return False
    return bool(re.sub(r"[\s0.,:+-]", "", text))


def classify_tag(group: str, tag: str) -> Optional[str]:
    lowered = tag.lower()
    if lowered.startswith("gps") or group.upper().startswith("GPS"):
        return None if lowered == "gpsversionid" else "gps"
    if "serialnumber" in lowered or lowered in ("serial", "cameraserialnumber", "internalserialnumber"):
        return "serial_numbers"
    if lowered in _NAME_TAGS:
        return "names"
    if lowered in _ID_TAGS:
        return "unique_ids"
    return None


def find_exiftool() -> Optional[str]:
    return shutil.which("exiftool")


def inspect_metadata(paths: Sequence[Union[str, Path]], *, use_exiftool: bool = True) -> MetadataReport:
    """Report metadata that publishing would expose, without modifying any file."""
    paths = [Path(p) for p in paths]
    exiftool = find_exiftool() if use_exiftool else None
    if exiftool:
        try:
            return _inspect_exiftool(exiftool, paths)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            report = _inspect_builtin(paths)
            report.tool = "the built-in EXIF reader (exiftool failed)"
            report.errors.append(f"exiftool failed ({exc}); used the built-in reader")
            return report
    report = _inspect_builtin(paths)
    if not use_exiftool:
        report.tool = "the built-in EXIF reader (exiftool not used)"
    return report


def _exiftool_json(exiftool: str, paths: Sequence[Path], arguments: Sequence[str]) -> List[dict]:
    records = []
    for start in range(0, len(paths), 64):
        chunk = [str(p) for p in paths[start:start + 64]]
        result = subprocess.run([exiftool, "-j", "-n", "-charset", "filename=utf8",
                                 *arguments, *chunk],
                                capture_output=True, timeout=600)
        if not result.stdout.strip():
            raise ValueError(result.stderr.decode("utf-8", "replace").strip() or "no output")
        records.extend(json.loads(result.stdout.decode("utf-8", "replace")))
    return records


def _inspect_exiftool(exiftool: str, paths: Sequence[Path]) -> MetadataReport:
    version = subprocess.run([exiftool, "-ver"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    report = MetadataReport(f"exiftool {version}".strip(), len(paths))
    for record in _exiftool_json(exiftool, paths, ["-a", "-G1", "-s"]):
        source = record.get("SourceFile", "?")
        if "Error" in record:
            report.errors.append(f"{Path(source).name}: {record['Error']}")
        for key, value in record.items():
            if ":" not in key:
                continue
            group, tag = key.split(":", 1)
            if group in ("System", "File", "ExifTool"):
                continue
            kind = classify_tag(group, tag)
            if kind and _meaningful(value):
                report.add(kind, f"{group}:{tag}", source, value if kind == "names" else None)
    report.not_checked.append("pixel content (for example visible labels, faces or text "
                              "in the image) is never checked")
    return report


# Built-in reader: standard EXIF/TIFF tags of JPEG and TIFF-based RAW files.
_TIFF_TAGS = {0x013B: ("names", "Artist"), 0x8298: ("names", "Copyright"),
              0x9C9D: ("names", "XPAuthor"), 0xA430: ("names", "CameraOwnerName"),
              0xA431: ("serial_numbers", "BodySerialNumber"),
              0xA435: ("serial_numbers", "LensSerialNumber"),
              0xA420: ("unique_ids", "ImageUniqueID")}
_GPS_NAMES = {1: "GPSLatitudeRef", 2: "GPSLatitude", 3: "GPSLongitudeRef", 4: "GPSLongitude",
              5: "GPSAltitudeRef", 6: "GPSAltitude", 7: "GPSTimeStamp", 9: "GPSStatus",
              18: "GPSMapDatum", 29: "GPSDateStamp"}
_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _read_ifd(handle, base, offset, order, limit=64 * 1024):
    entries = {}
    handle.seek(base + offset)
    raw = handle.read(2)
    if len(raw) < 2:
        return entries, 0
    count = struct.unpack(order + "H", raw)[0]
    if count > 4096:
        return entries, 0
    table = handle.read(12 * count + 4)
    if len(table) < 12 * count + 4:
        return entries, 0
    for index in range(count):
        tag, kind, number, value = struct.unpack(order + "HHI4s", table[12 * index:12 * index + 12])
        size = _TYPE_SIZES.get(kind, 1) * number
        if size > limit:
            continue
        if size <= 4:
            data = value[:size]
        else:
            position = handle.tell()
            handle.seek(base + struct.unpack(order + "I", value)[0])
            data = handle.read(size)
            handle.seek(position)
        entries[tag] = (kind, number, data)
    next_offset = struct.unpack(order + "I", table[-4:])[0]
    return entries, next_offset


def _tiff_value(entry, order):
    kind, number, data = entry
    if kind == 2:
        return data.split(b"\x00", 1)[0].decode("latin-1", "replace")
    if kind == 3:
        return list(struct.unpack(order + "H" * (len(data) // 2), data[:len(data) // 2 * 2]))
    if kind in (4, 9):
        return list(struct.unpack(order + "I" * (len(data) // 4), data[:len(data) // 4 * 4]))
    if kind in (5, 10):
        values = struct.unpack(order + "I" * (len(data) // 4), data[:len(data) // 4 * 4])
        return [values[i] / values[i + 1] if values[i + 1] else 0
                for i in range(0, len(values) - 1, 2)]
    if kind == 1 and len(data) and data[-1:] == b"\x00" and number > 2 and not data.startswith(b"\x00"):
        return data.decode("utf-16-le", "replace")  # XP* tags
    return data.hex() if data.strip(b"\x00 ") else ""


def _tiff_base(handle) -> Optional[Tuple[int, str]]:
    """(offset of the TIFF header, byte order) for a JPEG or TIFF-based file."""
    handle.seek(0)
    head = handle.read(4)
    if head[:2] in (b"II", b"MM"):
        return 0, "<" if head[:2] == b"II" else ">"
    if head[:2] != b"\xff\xd8":
        return None
    handle.seek(2)
    while True:
        marker = handle.read(4)
        if len(marker) < 4 or marker[0] != 0xFF or marker[1] in (0xDA, 0xD9):
            return None
        length = struct.unpack(">H", marker[2:])[0]
        start = handle.tell()
        if marker[1] == 0xE1 and handle.read(6) == b"Exif\x00\x00":
            order = handle.read(2)
            return start + 6, "<" if order == b"II" else ">"
        handle.seek(start + length - 2)


def _builtin_tiff(path: Path):
    """Tag findings and (width, height, bits) from standard TIFF/EXIF structures."""
    found = []
    shape = None
    with open(path, "rb") as handle:
        located = _tiff_base(handle)
        if located is None:
            return None, None
        base, order = located
        handle.seek(base + 4)
        offset = struct.unpack(order + "I", handle.read(4))[0]
        visited = set()
        sub_ifds = []
        while offset and offset not in visited and len(visited) < 16:
            visited.add(offset)
            entries, offset = _read_ifd(handle, base, offset, order)
            if shape is None and 0x0100 in entries and 0x0101 in entries:
                bits = _tiff_value(entries[0x0102], order) if 0x0102 in entries else [8]
                shape = (_tiff_value(entries[0x0100], order)[0],
                         _tiff_value(entries[0x0101], order)[0], (bits or [8])[0])
            for tag, (kind, name) in _TIFF_TAGS.items():
                if tag in entries and _meaningful(_tiff_value(entries[tag], order)):
                    found.append((kind, name, _tiff_value(entries[tag], order)))
            for pointer in (0x8769, 0x8825):
                if pointer in entries:
                    sub_ifds.append((pointer, _tiff_value(entries[pointer], order)[0]))
        for pointer, sub_offset in sub_ifds:
            entries, _ = _read_ifd(handle, base, sub_offset, order)
            if pointer == 0x8825:
                for tag, entry in entries.items():
                    if tag != 0 and _meaningful(_tiff_value(entry, order)):
                        found.append(("gps", _GPS_NAMES.get(tag, f"GPS tag 0x{tag:04x}"), None))
            else:
                for tag, (kind, name) in _TIFF_TAGS.items():
                    if tag in entries and _meaningful(_tiff_value(entries[tag], order)):
                        found.append((kind, name, _tiff_value(entries[tag], order)))
    return found, shape


def _inspect_builtin(paths: Sequence[Path]) -> MetadataReport:
    report = MetadataReport("the built-in EXIF reader (exiftool not found)", len(paths))
    report.not_checked = [
        "camera maker notes (vendor-specific serial numbers are usually stored there)",
        "XMP and IPTC fields (owner, creator and copyright names, sometimes GPS)",
        "pixel content (for example visible labels, faces or text in the image)",
    ]
    unreadable = []
    for path in paths:
        try:
            found, _shape = _builtin_tiff(path)
        except (OSError, struct.error, IndexError, ValueError) as exc:
            report.errors.append(f"{path.name}: {exc}")
            continue
        if found is None:
            unreadable.append(path.name)
            continue
        for kind, tag, value in found:
            report.add(kind, tag, str(path), value if kind == "names" else None)
    if unreadable:
        report.not_checked.append(
            f"{len(unreadable)} file(s) are not JPEG or TIFF-based, e.g. {unreadable[:3]}")
    return report


def image_shape(path: Union[str, Path], *, use_exiftool: bool = True) -> Optional[Tuple[int, int, int]]:
    """(width, height, bits per sample) of the main image, or None when unknown."""
    path = Path(path)
    exiftool = find_exiftool() if use_exiftool else None
    if exiftool:
        try:
            record = _exiftool_json(exiftool, [path], ["-ImageWidth", "-ImageHeight",
                                                       "-BitsPerSample", "-ValidBits"])[0]
            width, height = record.get("ImageWidth"), record.get("ImageHeight")
            bits = record.get("ValidBits") or record.get("BitsPerSample")
            bits = int(str(bits).split()[0]) if bits else 8
            if _is_int(width, 1) and _is_int(height, 1):
                return width, height, bits
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    try:
        with open(path, "rb") as handle:
            head = handle.read(32)
            if head.startswith(b"\x89PNG") and head[12:16] == b"IHDR":
                width, height = struct.unpack(">II", head[16:24])
                return width, height, head[24]
            if head.startswith(b"\xff\xd8"):
                handle.seek(2)
                while True:
                    marker = handle.read(4)
                    if len(marker) < 4 or marker[0] != 0xFF:
                        break
                    length = struct.unpack(">H", marker[2:])[0]
                    if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                        bits, height, width = struct.unpack(">BHH", handle.read(5))
                        return width, height, bits
                    handle.seek(length - 2, os.SEEK_CUR)
        _found, shape = _builtin_tiff(path)
        return shape
    except (OSError, struct.error, IndexError, ValueError):
        return None


# ---- packing -----------------------------------------------------------------------------

_FORMAT_NAMES = {".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}


def list_stack_files(folder: Union[str, Path]) -> List[Path]:
    """Image files directly inside ``folder`` in OpenFocus name order."""
    from utils.sorting import natural_sort_key
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def _tar_size(sizes: Iterable[int]) -> int:
    total = sum(512 + (size + 511) // 512 * 512 for size in sizes) + 1024
    return (total + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE * tarfile.RECORDSIZE


def plan_parts(sizes: Sequence[int], part_bytes: int = DEFAULT_PART_BYTES) -> List[List[int]]:
    """Group frame indices, in order, into tar parts no larger than ``part_bytes``."""
    if part_bytes >= MAX_ASSET_BYTES:
        raise SampleStackError("part size must be below 2 GiB")
    parts: List[List[int]] = []
    current: List[int] = []
    for index, size in enumerate(sizes):
        if _tar_size([size]) > part_bytes:
            raise SampleStackError(f"frame {index} ({size} bytes) does not fit in a {part_bytes}-byte part")
        if current and _tar_size([sizes[i] for i in current] + [size]) > part_bytes:
            parts.append(current)
            current = []
        current.append(index)
    if current:
        parts.append(current)
    return parts


def write_tar(path: Path, members: Sequence[Tuple[str, Path]]) -> None:
    """Deterministic uncompressed tar: fixed order, times, owners and modes."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as raw, tarfile.open(fileobj=raw, mode="w",
                                                        format=tarfile.PAX_FORMAT) as bundle:
            for name, source in members:
                info = tarfile.TarInfo(name)
                info.size = source.stat().st_size
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(source, "rb") as reader:
                    bundle.addfile(info, reader)
            bundle.close()
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pack_stack(files: Sequence[Union[str, Path]], stack_id: str, *,
               out_dir: Optional[Union[str, Path]] = None, local_only: bool = False,
               title: Optional[str] = None, description: str = "",
               useful_for: Sequence[str] = (), tags: Sequence[str] = (),
               license: str = PLACEHOLDER_PREFIX + ": choose an SPDX licence identifier",
               attribution: str = PLACEHOLDER_PREFIX + ": photographer name and credit line",
               base_url: Optional[str] = None, part_bytes: int = DEFAULT_PART_BYTES,
               file_format: Optional[str] = None, width: Optional[int] = None,
               height: Optional[int] = None, bit_depth: Optional[int] = None,
               recommended: Optional[dict] = None,
               allow_identifying_metadata: bool = False, use_exiftool: bool = True,
               metadata_report: Optional[MetadataReport] = None,
               progress: Optional[Callable[[str], None]] = None) -> Tuple[dict, MetadataReport]:
    """Hash files into a manifest entry and, unless ``local_only``, write tar parts.

    Files are stored unchanged (RAW files are never recompressed or rewritten).
    Raises IdentifyingMetadataError when GPS data or serial numbers are found,
    archives are requested and ``allow_identifying_metadata`` is false.
    Returns (entry, report).
    """
    say = progress or (lambda message: None)
    if not _ID_RE.match(stack_id or ""):
        raise SampleStackError(f"id {stack_id!r} must be 1-80 of a-z, 0-9, '.', '_' or '-'")
    paths = [Path(p).absolute() for p in files]
    if not paths:
        raise SampleStackError("no files to pack")
    for path in paths:
        if not path.is_file():
            raise SampleStackError(f"{path} is not a file")
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        raise SampleStackError("two files share a name; frames are stored by file name")
    if local_only and len({path.parent for path in paths}) != 1:
        raise SampleStackError("a local-only entry needs all frames in one folder")
    say(f"Checking metadata of {len(paths)} file(s)...")
    report = metadata_report or inspect_metadata(paths, use_exiftool=use_exiftool)
    # A local-only entry publishes nothing, so findings are reported but not refused.
    if report.blocking and not allow_identifying_metadata and not local_only:
        kinds = ", ".join(KIND_LABELS[k] for k in report.blocking)
        raise IdentifyingMetadataError(
            f"the files contain {kinds}. Publishing the files would expose them. Remove "
            "them with your own tools, or pass --allow-identifying-metadata to continue "
            "knowingly.", report)
    shape = image_shape(paths[0], use_exiftool=use_exiftool)
    width = width or (shape[0] if shape else None)
    height = height or (shape[1] if shape else None)
    bit_depth = bit_depth or (shape[2] if shape else None)
    if not (width and height and bit_depth):
        raise SampleStackError(f"could not read the image size or bit depth of {paths[0].name}; "
                               "give them (--width, --height, --bit-depth)")
    frames = []
    for index, path in enumerate(paths):
        say(f"Hashing {index + 1}/{len(paths)} {path.name}")
        frames.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    last = image_shape(paths[-1], use_exiftool=use_exiftool) if len(paths) > 1 else shape
    if shape and last and shape[:2] != last[:2]:
        say(f"Warning: the first and last frames differ in size ({shape[:2]} vs {last[:2]})")
    entry = {
        "id": stack_id,
        "title": title or stack_id,
        "description": description or f"{PLACEHOLDER_PREFIX}: describe the subject",
        "useful_for": list(useful_for),
        "tags": list(tags),
        "frame_count": len(frames),
        "format": file_format or _FORMAT_NAMES.get(paths[0].suffix.lower(),
                                                   paths[0].suffix.lstrip(".").upper()),
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "total_bytes": sum(f["bytes"] for f in frames),
        "license": license,
        "attribution": attribution,
    }
    if recommended:
        entry["recommended"] = recommended
    if local_only:
        entry["local_path"] = str(paths[0].parent)
    else:
        if out_dir is None:
            raise SampleStackError("an output folder is needed to write archive parts")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        groups = plan_parts([f["bytes"] for f in frames], part_bytes)
        archives = []
        for number, group in enumerate(groups, 1):
            name = f"{stack_id}.tar" if len(groups) == 1 else f"{stack_id}.part{number:02d}.tar"
            say(f"Writing {name} ({len(group)} frame(s))")
            write_tar(out / name, [(frames[i]["path"], paths[i]) for i in group])
            for i in group:
                frames[i]["archive"] = number - 1
            size = (out / name).stat().st_size
            if size >= MAX_ASSET_BYTES:
                raise SampleStackError(f"{name} is {size} bytes, not below 2 GiB")
            url = (base_url.rstrip("/") + "/" + urllib.parse.quote(name)) if base_url else name
            archives.append({"name": name, "url": url, "bytes": size, "sha256": sha256_file(out / name)})
        entry["archives"] = archives
    entry["frames"] = frames
    return entry, report


def subset_variant(parent: dict, files: Sequence[Union[str, Path]], variant_id: str, *,
                   title: Optional[str] = None, description: str = "",
                   useful_for: Sequence[str] = (), tags: Sequence[str] = (),
                   recommended: Optional[dict] = None, check_hashes: bool = True) -> dict:
    """A subset variant entry naming parent frames (verified by name, size and hash)."""
    by_name = {f["path"]: f for f in parent.get("frames", [])}
    names = []
    for item in files:
        path = Path(item)
        frame = by_name.get(path.name)
        if frame is None:
            raise SampleStackError(f"{path.name} is not a frame of {parent.get('id')!r}")
        if path.exists():
            if path.stat().st_size != frame["bytes"] or (
                    check_hashes and sha256_file(path) != frame["sha256"]):
                raise SampleStackError(f"{path} differs from the parent's frame {path.name}")
        names.append(path.name)
    variant = {"id": variant_id, "kind": "subset", "title": title or variant_id,
               "description": description or f"First {len(names)} frames of {parent.get('id')}",
               "frame_count": len(names), "frames": names}
    if useful_for:
        variant["useful_for"] = list(useful_for)
    if tags:
        variant["tags"] = list(tags)
    if recommended:
        variant["recommended"] = recommended
    return variant


def empty_manifest(title: str = "", description: str = "") -> dict:
    data = {"schema": SCHEMA}
    if title:
        data["title"] = title
    if description:
        data["description"] = description
    data["stacks"] = []
    return data


def merge_entry(manifest_data: dict, entry: dict, *, variant_of: Optional[str] = None) -> dict:
    """Insert or replace a stack (or a variant of ``variant_of``); returns the data."""
    stacks = manifest_data.setdefault("stacks", [])
    if variant_of is None:
        for index, existing in enumerate(stacks):
            if existing.get("id") == entry["id"]:
                if "variants" in existing and "variants" not in entry:
                    entry = dict(entry, variants=existing["variants"])
                stacks[index] = entry
                break
        else:
            stacks.append(entry)
        return manifest_data
    for existing in stacks:
        if existing.get("id") == variant_of:
            variants = existing.setdefault("variants", [])
            for index, old in enumerate(variants):
                if old.get("id") == entry["id"]:
                    variants[index] = entry
                    break
            else:
                variants.append(entry)
            return manifest_data
    raise SampleStackError(f"parent stack {variant_of!r} is not in the manifest")


def stack_summary(stack: Stack) -> dict:
    """Plain data describing a stack (for info/list output)."""
    data = {k: v for k, v in stack.data.items() if k not in ("frames", "archives", "variants")}
    data["frame_count"] = len(stack.frames)
    data["total_bytes"] = stack.total_bytes
    data["frames"] = [f.path for f in stack.frames]
    data["archives"] = [{"name": a.name, "url": a.url, "bytes": a.bytes, "sha256": a.sha256}
                        for a in stack.archives]
    if stack.local_path is not None:
        data["local_path"] = str(stack.local_path)
    if stack.parent:
        data["variant_of"] = stack.parent
        data["kind"] = stack.kind
    if stack.variant_ids:
        data["variants"] = list(stack.variant_ids)
    return data
