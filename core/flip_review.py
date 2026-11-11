"""Review sets and review records for the flip review viewer.

This module is the data side of the viewer and imports only the Python
standard library (plus ``constants`` for the version string). It must stay
free of Qt so the headless package and future batch runners can create,
validate and export reviews without the GUI.

Concepts
--------
*Review set*: the pictures being compared. It comes from a list of files, a
folder, or a review-set manifest (``openfocus/review-set`` JSON) that adds a
title, labels, free-form metadata and an optional reference candidate.

*Review record*: the reviewer's scores and notes for one review set
(``openfocus/flip-review`` JSON). Each candidate is identified by file name,
byte size, SHA-256 and pixel dimensions, so a record is never applied to
pictures it was not made for.

*Export*: a sanitized copy of a record meant for sharing. By default it holds
no pixels, no absolute paths and no reviewer name.
"""
from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

REVIEW_FORMAT = "openfocus/flip-review"
REVIEW_VERSION = 1
SET_FORMAT = "openfocus/review-set"
SET_VERSION = 1
EXPORT_FORMAT = "openfocus/flip-review-export"
EXPORT_VERSION = 1
TOOL_NAME = "OpenFocus flip review"

RATING_MIN = 1
RATING_MAX = 5
# Optional per-criterion ratings. The list is data: records store the
# criteria they were made with, and the viewer builds its panel from it.
CRITERIA = ("sharpness", "halos", "seams", "noise", "colour")

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jpe",
                    ".jfif", ".bmp", ".webp")
FOLDER_RECORD_NAME = "openfocus-review.json"
_HASH_CHUNK = 4 << 20


class ReviewError(ValueError):
    """A review set or record cannot be used as requested."""


class ManifestError(ReviewError):
    """A review-set manifest is malformed."""


class DimensionMismatch(ReviewError):
    """The candidates do not share one pixel size."""


class RecordMismatch(ReviewError):
    """A review record describes different pictures than the review set."""

    def __init__(self, path: str, problems: Sequence[str]):
        self.path = path
        self.problems = list(problems)
        super().__init__(
            f"Review record {os.path.basename(path)} does not match these "
            "images: " + "; ".join(self.problems))


def _app_version() -> str:
    try:
        from constants import APP_VERSION
        return str(APP_VERSION)
    except Exception:  # pragma: no cover - constants always ships
        return "unknown"


def utc_now() -> str:
    return (_datetime.datetime.now(_datetime.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))


def _natural_key(name: str):
    return [(0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", os.path.basename(name)) if part]


def blind_letter(position: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (spreadsheet-style neutral labels)."""
    letters = ""
    position += 1
    while position:
        position, rest = divmod(position - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


# ---------------------------------------------------------------------------
# Candidate identity
# ---------------------------------------------------------------------------

def file_sha256(path: str, cancel: Optional[Callable[[], bool]] = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            if cancel is not None and cancel():
                raise InterruptedError("hashing cancelled")
            chunk = stream.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateIdentity:
    name: str
    size: int
    sha256: str
    width: int
    height: int

    def same_pixels(self, other: "CandidateIdentity") -> bool:
        """Same file content and decoded size; the name may differ."""
        return (self.size == other.size and self.sha256 == other.sha256
                and self.width == other.width and self.height == other.height)

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "sha256": self.sha256,
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data) -> "CandidateIdentity":
        try:
            return cls(str(data["name"]), int(data["size"]), str(data["sha256"]).lower(),
                       int(data["width"]), int(data["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError(f"Invalid candidate identity: {exc}") from None

    def describe(self) -> str:
        return (f"{self.name} ({self.width}x{self.height}, {self.size} bytes, "
                f"sha256 {self.sha256[:12]})")


def identify(path: str, width: int, height: int, *,
             sha256: Optional[str] = None, cancel=None) -> CandidateIdentity:
    """Identity of a decoded candidate; ``width``/``height`` are display pixels."""
    return CandidateIdentity(os.path.basename(path), os.path.getsize(path),
                             sha256 or file_sha256(path, cancel), int(width), int(height))


def check_dimensions(items: Sequence[tuple]) -> None:
    """Raise DimensionMismatch unless every (name, width, height) agrees."""
    sizes: Dict[tuple, List[str]] = {}
    for name, width, height in items:
        sizes.setdefault((int(width), int(height)), []).append(str(name))
    if len(sizes) > 1:
        groups = "; ".join(f"{w}x{h}: {', '.join(names)}"
                           for (w, h), names in sizes.items())
        raise DimensionMismatch(
            f"The images do not have the same pixel dimensions ({groups})")


# ---------------------------------------------------------------------------
# Review sets
# ---------------------------------------------------------------------------

@dataclass
class SetCandidate:
    label: str
    path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ReviewSet:
    candidates: List[SetCandidate]
    title: str = ""
    reference: Optional[int] = None
    kind: str = "files"          # "files", "folder" or "manifest"
    source: Optional[str] = None  # folder or manifest path

    def __post_init__(self):
        if not self.candidates:
            raise ReviewError("A review set needs at least one image")
        labels = [candidate.label for candidate in self.candidates]
        if len(set(labels)) != len(labels):
            # Fall back to unique labels rather than refusing two files
            # that share a basename in different folders.
            seen: Dict[str, int] = {}
            for candidate in self.candidates:
                count = seen.get(candidate.label, 0)
                seen[candidate.label] = count + 1
                if count:
                    candidate.label = f"{candidate.label} ({count + 1})"
        if self.reference is not None and not 0 <= self.reference < len(self.candidates):
            raise ReviewError("Reference candidate is out of range")

    def default_record_path(self) -> str:
        """Where the review record lives unless ``--review`` overrides it."""
        if self.kind == "manifest" and self.source:
            stem = os.path.splitext(os.path.basename(self.source))[0]
            return os.path.join(os.path.dirname(os.path.abspath(self.source)),
                                f"{stem}.review.json")
        if self.kind == "folder" and self.source:
            return os.path.join(os.path.abspath(self.source), FOLDER_RECORD_NAME)
        folder = os.path.dirname(os.path.abspath(self.candidates[0].path))
        # Sorted so "a b" and "b a" reopen the same record.
        stems = sorted(os.path.splitext(os.path.basename(c.path))[0]
                       for c in self.candidates)
        name = re.sub(r"[^\w.-]+", "_", "_".join(stems)).strip("._") or "set"
        if len(name) > 64:
            digest = hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()[:10]
            name = f"{name[:40]}-{digest}"
        return os.path.join(folder, f"openfocus-review-{name}.json")


def _label_for(path: str) -> str:
    return os.path.basename(path)


def review_set_from_paths(paths: Sequence[str], *, title: str = "") -> ReviewSet:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise ReviewError("Image not found: " + ", ".join(missing))
    return ReviewSet([SetCandidate(_label_for(p), os.path.abspath(p)) for p in paths],
                     title=title, kind="files")


def review_set_from_folder(folder: str) -> ReviewSet:
    if not os.path.isdir(folder):
        raise ReviewError(f"Folder not found: {folder}")
    names = sorted((name for name in os.listdir(folder)
                    if name.lower().endswith(IMAGE_EXTENSIONS)
                    and os.path.isfile(os.path.join(folder, name))), key=_natural_key)
    if not names:
        raise ReviewError(f"No images found in {folder}")
    return ReviewSet([SetCandidate(name, os.path.abspath(os.path.join(folder, name)))
                      for name in names],
                     title=os.path.basename(os.path.abspath(folder)),
                     kind="folder", source=os.path.abspath(folder))


def _json_value(value, where):
    """Reject values JSON cannot round-trip (NaN, non-string keys)."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{where}: {exc}") from None
    return value


def parse_review_set(data, base_dir: str, *, source: Optional[str] = None) -> ReviewSet:
    """Validate a manifest dictionary; candidate paths resolve against base_dir."""
    if not isinstance(data, dict):
        raise ManifestError("A review-set manifest must be a JSON object")
    fmt = data.get("format", SET_FORMAT)
    if fmt != SET_FORMAT:
        raise ManifestError(f"Not a review-set manifest (format {fmt!r})")
    version = data.get("version", SET_VERSION)
    if not isinstance(version, int) or version < 1:
        raise ManifestError(f"Invalid manifest version {version!r}")
    if version > SET_VERSION:
        raise ManifestError(f"Manifest version {version} is newer than this "
                            f"viewer supports ({SET_VERSION})")
    title = data.get("title", "")
    if not isinstance(title, str):
        raise ManifestError("title must be a string")
    entries = data.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("candidates must be a non-empty list")
    candidates = []
    for number, entry in enumerate(entries, 1):
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) \
                or not entry["path"].strip():
            raise ManifestError(f"candidate {number} needs a path")
        path = entry["path"]
        resolved = os.path.normpath(path if os.path.isabs(path)
                                    else os.path.join(base_dir, path))
        label = entry.get("label", _label_for(path))
        if not isinstance(label, str) or not label.strip():
            raise ManifestError(f"candidate {number}: label must be a non-empty string")
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ManifestError(f"candidate {number}: metadata must be an object")
        candidates.append(SetCandidate(label.strip(), resolved,
                                       _json_value(metadata, f"candidate {number} metadata")))
    labels = [c.label for c in candidates]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ManifestError("Duplicate candidate labels: " + ", ".join(duplicates))
    reference = data.get("reference")
    if reference is not None:
        if isinstance(reference, bool):
            raise ManifestError("reference must be a label or an index")
        if isinstance(reference, int):
            if not 0 <= reference < len(candidates):
                raise ManifestError(f"reference index {reference} is out of range")
        elif isinstance(reference, str):
            if reference not in labels:
                raise ManifestError(f"reference {reference!r} is not a candidate label")
            reference = labels.index(reference)
        else:
            raise ManifestError("reference must be a label or an index")
    missing = [c.path for c in candidates if not os.path.isfile(c.path)]
    if missing:
        raise ManifestError("Image not found: " + ", ".join(missing))
    return ReviewSet(candidates, title=title, reference=reference,
                     kind="manifest", source=source)


def load_review_set(path: str) -> ReviewSet:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{os.path.basename(path)} is not valid JSON: {exc}") from None
    return parse_review_set(data, os.path.dirname(os.path.abspath(path)),
                            source=os.path.abspath(path))


def resolve_inputs(inputs: Sequence[str]) -> ReviewSet:
    """Files, one folder, or one ``.json`` manifest -> ReviewSet."""
    inputs = [os.fspath(item) for item in inputs]
    if not inputs:
        raise ReviewError("No images given")
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        return review_set_from_folder(inputs[0])
    if len(inputs) == 1 and inputs[0].lower().endswith(".json"):
        return load_review_set(inputs[0])
    folders = [item for item in inputs if os.path.isdir(item)]
    if folders:
        raise ReviewError("Give either one folder or a list of image files")
    return review_set_from_paths(inputs)


# ---------------------------------------------------------------------------
# Review record
# ---------------------------------------------------------------------------

def _check_rating(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) \
            or not RATING_MIN <= value <= RATING_MAX:
        raise ReviewError(f"Rating must be {RATING_MIN}-{RATING_MAX} or empty")
    return value


def _relative_path(path: str, record_path: str) -> str:
    """Path relative to the record when it is nearby, otherwise absolute."""
    try:
        relative = os.path.relpath(path, os.path.dirname(os.path.abspath(record_path)))
    except ValueError:  # another drive on Windows
        return os.path.abspath(path)
    parts = relative.replace("\\", "/").split("/")
    return os.path.abspath(path) if parts.count("..") > 2 else relative


def atomic_write_json(path: str, data) -> None:
    """Write beside the destination, flush to disk, then replace."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".flip-review-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class ReviewRecord:
    """Scores and notes for one review set, bound to its candidates.

    Candidates are addressed by stable record ids (``c1``, ``c2``...). The
    viewer maps its display positions to ids through ``order()``, so ratings
    made while blind always attach to the true candidate.
    """

    def __init__(self, data: dict, path: Optional[str]):
        self._data = data
        self.path = path
        self.dirty = False
        self.revision = 0  # incremented on every change
        self._by_id = {c["id"]: c for c in data["candidates"]}
        self._set_ids = [c["id"] for c in data["candidates"]]

    # -- creation, loading, binding ----------------------------------------

    @classmethod
    def new(cls, review_set: ReviewSet, identities: Sequence[CandidateIdentity],
            path: Optional[str] = None, *, criteria: Sequence[str] = CRITERIA) -> "ReviewRecord":
        if len(identities) != len(review_set.candidates):
            raise ReviewError("One identity per candidate is required")
        candidates = []
        for number, (candidate, identity) in enumerate(zip(review_set.candidates, identities), 1):
            candidates.append({
                "id": f"c{number}", "label": candidate.label,
                "file": identity.to_dict(),
                "path": _relative_path(candidate.path, path) if path else candidate.path,
                "metadata": dict(candidate.metadata),
                "rating": None, "criteria": {}, "comment": "", "rated_blind": False,
            })
        now = utc_now()
        data = {
            "format": REVIEW_FORMAT, "version": REVIEW_VERSION,
            "title": review_set.title, "reviewer": "",
            "created": now, "modified": now,
            "tool": {"name": TOOL_NAME, "version": _app_version()},
            "criteria": list(criteria),
            "candidates": candidates,
            "reference": (candidates[review_set.reference]["id"]
                          if review_set.reference is not None else None),
            "best": None, "set_comment": "", "pins": [], "next_pin": 1,
            "blind": {"enabled": False, "revealed": False, "order": []},
        }
        return cls(data, path)

    @classmethod
    def read(cls, path: str) -> "ReviewRecord":
        """Parse and validate a record file without binding it."""
        try:
            with open(path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ReviewError(f"{os.path.basename(path)} is not valid JSON: {exc}") from None
        return cls(cls._validate(data, path), path)

    @staticmethod
    def _validate(data, path) -> dict:
        name = os.path.basename(path)
        if not isinstance(data, dict) or data.get("format") != REVIEW_FORMAT:
            raise ReviewError(f"{name} is not an OpenFocus flip review record")
        version = data.get("version")
        if not isinstance(version, int) or version < 1:
            raise ReviewError(f"{name}: invalid record version {version!r}")
        if version > REVIEW_VERSION:
            raise ReviewError(f"{name} was written by a newer viewer (record version "
                              f"{version}, supported {REVIEW_VERSION})")
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ReviewError(f"{name}: no candidates")
        ids = set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
                raise ReviewError(f"{name}: malformed candidate entry")
            if candidate["id"] in ids:
                raise ReviewError(f"{name}: duplicate candidate id {candidate['id']}")
            ids.add(candidate["id"])
            CandidateIdentity.from_dict(candidate.get("file"))
            candidate["rating"] = _check_rating(candidate.get("rating"))
            criteria = candidate.get("criteria") or {}
            if not isinstance(criteria, dict):
                raise ReviewError(f"{name}: criteria must be an object")
            candidate["criteria"] = {str(k): _check_rating(v) for k, v in criteria.items()
                                     if v is not None}
            candidate.setdefault("label", candidate["file"]["name"])
            candidate.setdefault("metadata", {})
            candidate["comment"] = str(candidate.get("comment") or "")
            candidate["rated_blind"] = bool(candidate.get("rated_blind", False))
        data.setdefault("title", "")
        data["reviewer"] = str(data.get("reviewer") or "")
        data.setdefault("tool", {"name": TOOL_NAME, "version": _app_version()})
        data["criteria"] = [str(c) for c in data.get("criteria") or CRITERIA]
        if data.get("best") is not None and data["best"] not in ids:
            raise ReviewError(f"{name}: best mark names an unknown candidate")
        if data.get("reference") is not None and data["reference"] not in ids:
            data["reference"] = None
        data["set_comment"] = str(data.get("set_comment") or "")
        pins = data.get("pins") or []
        if not isinstance(pins, list):
            raise ReviewError(f"{name}: pins must be a list")
        for pin in pins:
            if (not isinstance(pin, dict) or not isinstance(pin.get("id"), str)
                    or isinstance(pin.get("x"), bool) or not isinstance(pin.get("x"), int)
                    or isinstance(pin.get("y"), bool) or not isinstance(pin.get("y"), int)):
                raise ReviewError(f"{name}: malformed pin")
            if pin.get("candidate") is not None and pin["candidate"] not in ids:
                raise ReviewError(f"{name}: pin {pin['id']} names an unknown candidate")
            pin["text"] = str(pin.get("text") or "")
        data["pins"] = pins
        numbers = [int(p["id"][1:]) for p in pins if re.fullmatch(r"p\d+", p["id"])]
        data["next_pin"] = max([int(data.get("next_pin") or 1), *(n + 1 for n in numbers)])
        blind = data.get("blind") or {}
        order = [cid for cid in blind.get("order") or [] if cid in ids]
        if blind.get("enabled") and sorted(order) != sorted(ids):
            raise ReviewError(f"{name}: blind order does not cover every candidate")
        data["blind"] = {"enabled": bool(blind.get("enabled")),
                         "revealed": bool(blind.get("revealed")), "order": order}
        return data

    def bind(self, review_set: ReviewSet, identities: Sequence[CandidateIdentity]) -> None:
        """Match set candidates to record candidates or raise RecordMismatch.

        Matching uses size, SHA-256 and pixel dimensions; a renamed file with
        the same content still matches (preferring equal names when two
        candidates have identical content). Any unmatched picture on either
        side is a mismatch: the record is never applied partially.
        """
        unmatched = list(self._data["candidates"])
        ids: List[Optional[str]] = []
        problems = []
        reported = set()
        for candidate, identity in zip(review_set.candidates, identities):
            options = [c for c in unmatched
                       if CandidateIdentity.from_dict(c["file"]).same_pixels(identity)]
            options.sort(key=lambda c: c["file"]["name"] != identity.name)
            if options:
                unmatched.remove(options[0])
                ids.append(options[0]["id"])
                continue
            ids.append(None)
            same_name = [c for c in unmatched if c["file"]["name"] == identity.name]
            if same_name:
                reported.add(same_name[0]["id"])
                recorded = CandidateIdentity.from_dict(same_name[0]["file"])
                problems.append(f"{identity.name} has changed (recorded "
                                f"{recorded.describe()}, now {identity.describe()})")
            else:
                problems.append(f"{identity.describe()} is not in the record")
        for candidate in unmatched:
            identity = CandidateIdentity.from_dict(candidate["file"])
            if candidate["id"] not in reported:
                problems.append(f"recorded {identity.describe()} is not in this set")
        if problems:
            raise RecordMismatch(self.path or "record", problems)
        self._set_ids = ids
        # Labels and metadata follow the current set; this is not a change
        # worth saving by itself.
        for candidate, cid in zip(review_set.candidates, ids):
            entry = self._by_id[cid]
            entry["label"] = candidate.label
            if candidate.metadata:
                entry["metadata"] = dict(candidate.metadata)
            if self.path:
                entry["path"] = _relative_path(candidate.path, self.path)
            entry["file"]["name"] = os.path.basename(candidate.path)
        if review_set.reference is not None:
            self._data["reference"] = ids[review_set.reference]

    @classmethod
    def open(cls, path: str, review_set: ReviewSet,
             identities: Sequence[CandidateIdentity]) -> "ReviewRecord":
        """Load and bind an existing record, or start a new one at ``path``."""
        if not os.path.exists(path):
            return cls.new(review_set, identities, path)
        record = cls.read(path)
        record.bind(review_set, identities)
        return record

    # -- accessors ---------------------------------------------------------

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._data))

    @property
    def set_ids(self) -> List[str]:
        """Record ids in review-set order."""
        return list(self._set_ids)

    @property
    def criteria(self) -> List[str]:
        return list(self._data["criteria"])

    def candidate(self, cid: str) -> dict:
        return self._by_id[cid]

    def label(self, cid: str) -> str:
        return self._by_id[cid]["label"]

    def rating(self, cid: str) -> Optional[int]:
        return self._by_id[cid]["rating"]

    def criterion(self, cid: str, name: str) -> Optional[int]:
        return self._by_id[cid]["criteria"].get(name)

    def comment(self, cid: str) -> str:
        return self._by_id[cid]["comment"]

    @property
    def best(self) -> Optional[str]:
        return self._data["best"]

    @property
    def reference(self) -> Optional[str]:
        return self._data["reference"]

    @property
    def overall_comment(self) -> str:
        """Comment on the set as a whole."""
        return self._data["set_comment"]

    @property
    def reviewer(self) -> str:
        return self._data["reviewer"]

    @property
    def pins(self) -> List[dict]:
        return [dict(pin) for pin in self._data["pins"]]

    def pin(self, pid: str) -> dict:
        for pin in self._data["pins"]:
            if pin["id"] == pid:
                return dict(pin)
        raise KeyError(pid)

    # -- blind mode --------------------------------------------------------

    @property
    def blind_active(self) -> bool:
        blind = self._data["blind"]
        return blind["enabled"] and not blind["revealed"]

    @property
    def blind_used(self) -> bool:
        return self._data["blind"]["enabled"]

    def order(self) -> List[str]:
        """Candidate ids in viewing order: shuffled while blind."""
        if self.blind_active:
            return list(self._data["blind"]["order"])
        return self.set_ids

    def display_label(self, cid: str) -> str:
        if self.blind_active:
            return blind_letter(self._data["blind"]["order"].index(cid))
        return self.label(cid)

    def color_index(self, cid: str) -> int:
        """Stable colour slot: the blind position once a blind order exists,
        so colours follow the neutral letters and never identify a picture."""
        order = self._data["blind"]["order"]
        if order and cid in order:
            return order.index(cid)
        return self.set_ids.index(cid)

    def start_blind(self, rng: Optional[random.Random] = None) -> None:
        """Shuffle the viewing order and hide labels until ``reveal()``."""
        order = self.set_ids
        (rng or random.SystemRandom()).shuffle(order)
        self._data["blind"] = {"enabled": True, "revealed": False, "order": order}
        self._changed()

    def reveal(self) -> None:
        if self._data["blind"]["enabled"] and not self._data["blind"]["revealed"]:
            self._data["blind"]["revealed"] = True
            self._changed()

    def stop_blind(self) -> None:
        """Leave blind mode before any reveal (ratings keep their flag)."""
        if self._data["blind"]["enabled"]:
            self._data["blind"] = {"enabled": False, "revealed": False, "order": []}
            self._changed()

    # -- mutations ---------------------------------------------------------

    def _changed(self) -> None:
        self.dirty = True
        self.revision += 1
        self._data["modified"] = utc_now()

    def set_rating(self, cid: str, value: Optional[int]) -> None:
        entry = self._by_id[cid]
        value = _check_rating(value)
        if entry["rating"] != value:
            entry["rating"] = value
            if value is not None:
                entry["rated_blind"] = self.blind_active
            self._changed()

    def set_criterion(self, cid: str, name: str, value: Optional[int]) -> None:
        entry = self._by_id[cid]["criteria"]
        value = _check_rating(value)
        if entry.get(name) != value:
            if value is None:
                entry.pop(name, None)
            else:
                entry[name] = value
            self._changed()

    def set_comment(self, cid: str, text: str) -> None:
        if self._by_id[cid]["comment"] != text:
            self._by_id[cid]["comment"] = str(text)
            self._changed()

    def set_best(self, cid: Optional[str]) -> None:
        """One best mark per set; ``None`` clears it."""
        if cid is not None and cid not in self._by_id:
            raise KeyError(cid)
        if self._data["best"] != cid:
            self._data["best"] = cid
            self._changed()

    def toggle_best(self, cid: str) -> None:
        self.set_best(None if self.best == cid else cid)

    def set_reference(self, cid: Optional[str]) -> None:
        if cid is not None and cid not in self._by_id:
            raise KeyError(cid)
        if self._data["reference"] != cid:
            self._data["reference"] = cid
            self._changed()

    def set_overall_comment(self, text: str) -> None:
        if self._data["set_comment"] != text:
            self._data["set_comment"] = str(text)
            self._changed()

    def set_reviewer(self, name: str) -> None:
        name = str(name).strip()
        if self._data["reviewer"] != name:
            self._data["reviewer"] = name
            self._changed()

    def add_pin(self, x: int, y: int, cid: Optional[str], text: str = "") -> str:
        file = self._data["candidates"][0]["file"]
        if not (0 <= int(x) < file["width"] and 0 <= int(y) < file["height"]):
            raise ReviewError("Pin position is outside the image")
        if cid is not None and cid not in self._by_id:
            raise KeyError(cid)
        pid = f"p{self._data['next_pin']}"
        self._data["next_pin"] += 1
        self._data["pins"].append({"id": pid, "x": int(x), "y": int(y),
                                   "candidate": cid, "text": str(text),
                                   "created": utc_now()})
        self._changed()
        return pid

    def update_pin(self, pid: str, *, text: Optional[str] = None,
                   candidate: Optional[str] = None) -> None:
        for pin in self._data["pins"]:
            if pin["id"] == pid:
                changed = False
                if text is not None and pin["text"] != text:
                    pin["text"] = str(text)
                    changed = True
                if candidate is not None and pin["candidate"] != candidate:
                    if candidate not in self._by_id:
                        raise KeyError(candidate)
                    pin["candidate"] = candidate
                    changed = True
                if changed:
                    self._changed()
                return
        raise KeyError(pid)

    def remove_pin(self, pid: str) -> None:
        before = len(self._data["pins"])
        self._data["pins"] = [pin for pin in self._data["pins"] if pin["id"] != pid]
        if len(self._data["pins"]) == before:
            raise KeyError(pid)
        self._changed()

    # -- persistence -------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ReviewError("No review record path")
        atomic_write_json(path, self._data)
        self.path = path
        self.dirty = False
        return path

    def export_dict(self, *, include_paths: bool = False,
                    include_reviewer: bool = False, base_dir: Optional[str] = None) -> dict:
        """A sanitized summary for sharing: no pixels, paths or names by default.

        ``include_paths`` adds each candidate's path as recorded (absolute
        when ``base_dir`` is given). Free-form metadata and texts have the
        home folder replaced by ``~`` and absolute paths in metadata reduced
        to their file name unless paths are requested.
        """
        clean = (lambda value: value) if include_paths else _scrub
        best = self.best
        candidates = []
        for cid in self.set_ids:
            entry = self._by_id[cid]
            item = {
                "label": entry["label"],
                "file": dict(entry["file"]),
                "metadata": clean(entry.get("metadata") or {}),
                "rating": entry["rating"],
                "criteria": {k: v for k, v in entry["criteria"].items()},
                "comment": _scrub_text(entry["comment"]),
                "best": cid == best,
                "rated_blind": entry["rated_blind"],
            }
            if include_paths:
                path = entry.get("path", "")
                if base_dir and path and not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(base_dir, path))
                item["path"] = path
            candidates.append(item)
        labels = {cid: self._by_id[cid]["label"] for cid in self._by_id}
        result = {
            "format": EXPORT_FORMAT, "version": EXPORT_VERSION,
            "title": _scrub_text(self._data["title"]),
            "exported": utc_now(),
            "created": self._data["created"], "modified": self._data["modified"],
            "tool": dict(self._data["tool"], exported_with=_app_version()),
            "criteria": self.criteria,
            "blind": {"used": self.blind_used, "revealed": self._data["blind"]["revealed"]},
            "reference": labels.get(self.reference),
            "best": labels.get(best),
            "set_comment": _scrub_text(self.overall_comment),
            "candidates": candidates,
            "pins": [{"x": pin["x"], "y": pin["y"],
                      "candidate": labels.get(pin["candidate"]),
                      "text": _scrub_text(pin["text"])} for pin in self._data["pins"]],
        }
        if include_reviewer and self.reviewer:
            result["reviewer"] = self.reviewer
        return result

    def export(self, path: str, **options) -> str:
        """Write the export as JSON, or as Markdown when ``path`` ends in .md."""
        if "base_dir" not in options and self.path:
            options["base_dir"] = os.path.dirname(os.path.abspath(self.path))
        data = self.export_dict(**options)
        if path.lower().endswith((".md", ".markdown")):
            directory = os.path.dirname(os.path.abspath(path))
            fd, temporary = tempfile.mkstemp(prefix=".flip-review-", suffix=".tmp",
                                             dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(export_markdown(data))
                os.replace(temporary, path)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        else:
            atomic_write_json(path, data)
        return path


def start_new_record(path: str, review_set: ReviewSet,
                     identities: Sequence[CandidateIdentity]) -> ReviewRecord:
    """Keep a mismatching record as a backup and start a fresh one at ``path``."""
    if os.path.exists(path):
        stamp = utc_now().replace(":", "").replace("-", "")
        stem, ext = os.path.splitext(path)
        backup = f"{stem}.mismatch-{stamp}{ext or '.json'}"
        os.replace(path, backup)
    return ReviewRecord.new(review_set, identities, path)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")


def _home_folders() -> List[str]:
    homes = {os.path.expanduser("~"), os.environ.get("USERPROFILE", ""),
             os.environ.get("HOME", "")}
    return sorted((h for h in homes if h and len(h) > 1), key=len, reverse=True)


def _scrub_text(text: str) -> str:
    for home in _home_folders():
        text = text.replace(home, "~")
    return text


def _scrub(value):
    """Reduce absolute paths in metadata to file names; hide the home folder."""
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("/", "~")) or _WINDOWS_ABSOLUTE.match(stripped):
            return os.path.basename(stripped.replace("\\", "/").rstrip("/")) or ""
        return _scrub_text(value)
    return value


def _md(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def export_markdown(data: dict) -> str:
    """Render an export dictionary as Markdown for an issue comment."""
    lines = [f"## Flip review: {_md(data['title'] or 'untitled set')}", ""]
    stars = lambda value: "-" if value is None else f"{value}/5"
    criteria = data["criteria"]
    header = ["Candidate", "Rating", "Best", *criteria, "Comment"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for item in data["candidates"]:
        row = [_md(item["label"]), stars(item["rating"]), "yes" if item["best"] else "",
               *(stars(item["criteria"].get(c)) for c in criteria), _md(item["comment"])]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if data.get("set_comment"):
        lines += ["**Set comment:** " + data["set_comment"].replace("\n", "  \n"), ""]
    if data["pins"]:
        lines.append("**Pinned notes** (image pixel coordinates)")
        lines.append("")
        for pin in data["pins"]:
            lines.append(f"- ({pin['x']}, {pin['y']}) {_md(pin['candidate'] or 'set')}: "
                         f"{_md(pin['text'])}")
        lines.append("")
    blind = data["blind"]
    if blind["used"]:
        lines += [f"Rated blind: {'revealed' if blind['revealed'] else 'not revealed'}.", ""]
    lines.append("<details><summary>Candidate files</summary>")
    lines.append("")
    for item in data["candidates"]:
        file = item["file"]
        extra = f", metadata `{json.dumps(item['metadata'], ensure_ascii=False)}`" \
            if item["metadata"] else ""
        path = f", path `{item['path']}`" if item.get("path") else ""
        lines.append(f"- {_md(item['label'])}: `{file['name']}` {file['width']}x"
                     f"{file['height']}, {file['size']} bytes, sha256 `{file['sha256']}`"
                     f"{path}{extra}")
    lines += ["", "</details>", "",
              f"_{data['tool'].get('name', TOOL_NAME)} {data['tool'].get('version', '')}, "
              f"exported {data['exported']}_", ""]
    return "\n".join(lines)

