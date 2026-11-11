"""Review sets, review records and exports of the flip review viewer (no Qt)."""
import json
import os
import random
import subprocess
import sys

import pytest

from core import flip_review as model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write(path, content: bytes):
    path.write_bytes(content)
    return str(path)


def _set_and_identities(tmp_path, contents=(b"first", b"second"), size=(40, 30)):
    paths = [_write(tmp_path / f"cand{i}.png", data) for i, data in enumerate(contents)]
    review_set = model.review_set_from_paths(paths)
    identities = [model.identify(p, *size) for p in paths]
    return review_set, identities


def test_model_imports_without_qt_or_numpy():
    code = ("import sys; import core.flip_review; "
            "loaded = {'PyQt6', 'numpy', 'cv2'} & {m.split('.')[0] for m in sys.modules}; "
            "assert not loaded, loaded")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_record_round_trip(tmp_path):
    review_set, identities = _set_and_identities(tmp_path)
    path = str(tmp_path / "review.json")
    record = model.ReviewRecord.open(path, review_set, identities)
    first, second = record.set_ids
    record.set_rating(first, 4)
    record.set_criterion(first, "halos", 2)
    record.set_comment(second, "seam at the left edge")
    record.set_best(second)
    record.set_overall_comment("both soft at the top")
    record.set_reviewer("someone")
    pin = record.add_pin(10, 5, second, "halo")
    assert record.dirty
    record.save()
    assert not record.dirty
    on_disk = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert on_disk["format"] == model.REVIEW_FORMAT and on_disk["version"] == 1
    assert on_disk["tool"]["version"] and on_disk["created"] and on_disk["modified"]
    assert on_disk["candidates"][0]["path"] == "cand0.png"  # relative to the record

    again = model.ReviewRecord.open(path, review_set, identities)
    assert again.rating(first) == 4
    assert again.criterion(first, "halos") == 2
    assert again.criterion(first, "noise") is None
    assert again.comment(second) == "seam at the left edge"
    assert again.best == second
    assert again.overall_comment == "both soft at the top"
    assert again.reviewer == "someone"
    assert (again.pin(pin)["x"], again.pin(pin)["y"], again.pin(pin)["candidate"],
            again.pin(pin)["text"]) == (10, 5, second, "halo")
    # A second pin never reuses an id.
    assert again.add_pin(1, 1, None) == "p2"


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path, monkeypatch):
    review_set, identities = _set_and_identities(tmp_path)
    path = tmp_path / "review.json"
    record = model.ReviewRecord.open(str(path), review_set, identities)
    record.set_rating(record.set_ids[0], 3)
    record.save()
    before = path.read_bytes()

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(model.json, "dump", broken_dump)
    record.set_rating(record.set_ids[0], 5)
    with pytest.raises(OSError):
        record.save()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cand0.png", "cand1.png", "review.json"]


def test_order_of_inputs_does_not_matter_and_renamed_copy_still_matches(tmp_path):
    review_set, identities = _set_and_identities(tmp_path)
    path = str(tmp_path / "review.json")
    record = model.ReviewRecord.open(path, review_set, identities)
    record.set_rating(record.set_ids[1], 2)
    record.save()

    renamed = tmp_path / "renamed.png"
    renamed.write_bytes(b"second")
    swapped = model.review_set_from_paths([str(renamed), review_set.candidates[0].path])
    swapped_ids = [model.identify(c.path, 40, 30) for c in swapped.candidates]
    again = model.ReviewRecord.open(path, swapped, swapped_ids)
    assert again.rating(again.set_ids[0]) == 2
    assert again.label(again.set_ids[0]) == "renamed.png"


@pytest.mark.parametrize("change", ["content", "dimensions", "extra", "missing"])
def test_record_is_never_applied_to_different_pictures(tmp_path, change):
    review_set, identities = _set_and_identities(tmp_path)
    path = str(tmp_path / "review.json")
    record = model.ReviewRecord.open(path, review_set, identities)
    record.set_rating(record.set_ids[0], 5)
    record.save()

    if change == "content":
        (tmp_path / "cand1.png").write_bytes(b"edited")
        identities = [model.identify(c.path, 40, 30) for c in review_set.candidates]
        expected = "cand1.png has changed"
    elif change == "dimensions":
        identities = [identities[0], model.identify(review_set.candidates[1].path, 41, 30)]
        expected = "cand1.png has changed"
    elif change == "extra":
        third = _write(tmp_path / "cand2.png", b"third")
        review_set = model.review_set_from_paths([*(c.path for c in review_set.candidates), third])
        identities = [*identities, model.identify(third, 40, 30)]
        expected = "cand2.png"
    else:
        review_set = model.review_set_from_paths([review_set.candidates[0].path])
        identities = identities[:1]
        expected = "recorded cand1.png"
    with pytest.raises(model.RecordMismatch) as caught:
        model.ReviewRecord.open(path, review_set, identities)
    assert any(expected in problem for problem in caught.value.problems), caught.value.problems
    assert "does not match" in str(caught.value)

    fresh = model.start_new_record(path, review_set, identities)
    assert fresh.rating(fresh.set_ids[0]) is None
    backups = [p.name for p in tmp_path.iterdir() if ".mismatch-" in p.name]
    assert len(backups) == 1


def test_newer_or_foreign_records_are_refused(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    with pytest.raises(model.ReviewError, match="not an OpenFocus flip review"):
        model.ReviewRecord.read(str(path))
    review_set, identities = _set_and_identities(tmp_path)
    data = model.ReviewRecord.new(review_set, identities).to_dict()
    data["version"] = model.REVIEW_VERSION + 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(model.ReviewError, match="newer"):
        model.ReviewRecord.read(str(path))


def test_ratings_are_validated_and_best_is_unique(tmp_path):
    review_set, identities = _set_and_identities(tmp_path)
    record = model.ReviewRecord.new(review_set, identities)
    first, second = record.set_ids
    for bad in (0, 6, True, 2.5):
        with pytest.raises(model.ReviewError):
            record.set_rating(first, bad)
    record.set_best(first)
    record.toggle_best(second)
    assert record.best == second
    record.toggle_best(second)
    assert record.best is None
    with pytest.raises(model.ReviewError):
        record.add_pin(40, 0, first)  # outside a 40 px wide image


def test_blind_shuffle_attaches_ratings_to_true_candidates(tmp_path):
    contents = [f"candidate {i}".encode() for i in range(6)]
    review_set, identities = _set_and_identities(tmp_path, contents)
    path = str(tmp_path / "review.json")
    record = model.ReviewRecord.open(path, review_set, identities)
    record.start_blind(random.Random(3))
    order = record.order()
    assert sorted(order) == sorted(record.set_ids) and order != record.set_ids
    assert [record.display_label(cid) for cid in order] == list("ABCDEF")
    # Rate the candidate shown as "C" while blind.
    shown_c = order[2]
    record.set_rating(shown_c, 5)
    record.save()

    again = model.ReviewRecord.open(path, review_set, identities)
    assert again.blind_active and again.order() == order
    assert again.display_label(shown_c) == "C"
    again.reveal()
    assert not again.blind_active
    assert again.order() == again.set_ids
    assert again.display_label(shown_c) == again.label(shown_c)
    assert again.rating(shown_c) == 5
    assert again.candidate(shown_c)["rated_blind"] is True
    true_index = again.set_ids.index(shown_c)
    assert again.label(shown_c) == f"cand{true_index}.png"


def test_blind_letters_continue_past_z():
    assert [model.blind_letter(i) for i in (0, 25, 26, 27, 51, 52)] == \
        ["A", "Z", "AA", "AB", "AZ", "BA"]


def test_export_privacy_defaults(tmp_path, monkeypatch):
    home = str(tmp_path / "home" / "person")
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    folder = tmp_path / "home" / "person" / "renders"
    folder.mkdir(parents=True)
    manifest = {
        "format": model.SET_FORMAT, "version": 1, "title": "Overlap test",
        "candidates": [
            {"label": "overlap 32", "path": "a.png",
             "metadata": {"revision": "abc123", "source": f"{home}/stacks/bug",
                          "parameters": {"tile_overlap": 32}}},
            {"label": "overlap 64", "path": "b.png"},
        ]}
    (folder / "a.png").write_bytes(b"a")
    (folder / "b.png").write_bytes(b"b")
    (folder / "set.json").write_text(json.dumps(manifest), encoding="utf-8")
    review_set = model.load_review_set(str(folder / "set.json"))
    identities = [model.identify(c.path, 8, 8) for c in review_set.candidates]
    record = model.ReviewRecord.open(review_set.default_record_path(), review_set, identities)
    first, second = record.set_ids
    record.set_reviewer("Reviewer Name")
    record.set_rating(first, 4)
    record.set_comment(first, f"compare with {home}/other.tif")
    record.add_pin(3, 4, second, "seam")
    record.save()

    out = tmp_path / "export.json"
    record.export(str(out))
    exported = out.read_text(encoding="utf-8")
    data = json.loads(exported)
    assert data["format"] == model.EXPORT_FORMAT
    assert home not in exported and "Reviewer Name" not in exported
    assert "reviewer" not in data
    assert all("path" not in item for item in data["candidates"])
    assert data["candidates"][0]["metadata"]["source"] == "bug"
    assert data["candidates"][0]["metadata"]["parameters"] == {"tile_overlap": 32}
    assert data["candidates"][0]["comment"] == "compare with ~/other.tif"
    assert data["candidates"][0]["file"]["sha256"] == identities[0].sha256
    assert data["pins"] == [{"x": 3, "y": 4, "candidate": "overlap 64", "text": "seam"}]
    assert data["title"] == "Overlap test" and data["candidates"][0]["rating"] == 4

    with_paths = json.loads(json.dumps(record.export_dict(
        include_paths=True, include_reviewer=True, base_dir=str(folder))))
    assert with_paths["reviewer"] == "Reviewer Name"
    assert with_paths["candidates"][0]["path"] == str(folder / "a.png")

    markdown = tmp_path / "export.md"
    record.export(str(markdown))
    text = markdown.read_text(encoding="utf-8")
    assert "| overlap 32 | 4/5 |" in text and home not in text


def test_manifest_parsing(tmp_path):
    (tmp_path / "renders").mkdir()
    for name in ("x.tif", "y.tif"):
        (tmp_path / "renders" / name).write_bytes(name.encode())
    manifest = tmp_path / "set.json"
    manifest.write_text(json.dumps({
        "format": "openfocus/review-set", "version": 1, "title": "T",
        "reference": "y.tif",
        "candidates": [{"label": "old", "path": "renders/x.tif",
                        "metadata": {"timings": {"render_s": 4.5}}},
                       "renders/y.tif"]}), encoding="utf-8")
    review_set = model.resolve_inputs([str(manifest)])
    assert review_set.kind == "manifest" and review_set.title == "T"
    assert review_set.reference == 1
    assert [c.label for c in review_set.candidates] == ["old", "y.tif"]
    assert review_set.candidates[0].path == str(tmp_path / "renders" / "x.tif")
    assert review_set.candidates[0].metadata == {"timings": {"render_s": 4.5}}
    assert review_set.default_record_path() == str(tmp_path / "set.review.json")


@pytest.mark.parametrize("manifest, message", [
    ({"format": "other", "candidates": ["a.png"]}, "Not a review-set"),
    ({"version": 9, "candidates": ["a.png"]}, "newer"),
    ({"candidates": []}, "non-empty"),
    ({"candidates": [{"label": "x"}]}, "needs a path"),
    ({"candidates": [{"path": "a.png", "label": "same"}, {"path": "b.png", "label": "same"}]},
     "Duplicate"),
    ({"candidates": ["a.png"], "reference": "missing"}, "not a candidate"),
    ({"candidates": ["a.png"], "reference": 4}, "out of range"),
    ({"candidates": ["absent.png"]}, "not found"),
    ({"candidates": [{"path": "a.png", "metadata": []}]}, "metadata"),
])
def test_manifest_errors(tmp_path, manifest, message):
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    with pytest.raises(model.ManifestError, match=message):
        model.parse_review_set(manifest, str(tmp_path))


def test_folder_and_file_inputs(tmp_path):
    for name in ("img10.png", "img2.png", "notes.txt", "img1.tif"):
        (tmp_path / name).write_bytes(name.encode())
    review_set = model.resolve_inputs([str(tmp_path)])
    assert [c.label for c in review_set.candidates] == ["img1.tif", "img2.png", "img10.png"]
    assert review_set.default_record_path() == str(tmp_path / model.FOLDER_RECORD_NAME)
    files = model.resolve_inputs([str(tmp_path / "img2.png"), str(tmp_path / "img1.tif")])
    reversed_files = model.resolve_inputs([str(tmp_path / "img1.tif"), str(tmp_path / "img2.png")])
    assert files.default_record_path() == reversed_files.default_record_path()
    assert os.path.basename(files.default_record_path()) == "openfocus-review-img1_img2.json"
    with pytest.raises(model.ReviewError, match="not found"):
        model.resolve_inputs([str(tmp_path / "absent.png")])


def test_dimension_mismatch_names_the_files():
    model.check_dimensions([("a.tif", 10, 20), ("b.tif", 10, 20)])
    with pytest.raises(model.DimensionMismatch) as caught:
        model.check_dimensions([("a.tif", 10, 20), ("b.tif", 10, 20), ("c.tif", 20, 10)])
    assert "a.tif, b.tif" in str(caught.value) and "20x10: c.tif" in str(caught.value)


def test_command_line_export_and_errors(tmp_path, capsys):
    import cv2
    import numpy as np
    from tools import flip_review as tool

    paths = []
    for index in range(2):
        path = tmp_path / f"r{index}.png"
        cv2.imwrite(str(path), np.full((12, 16, 3), index * 40, np.uint8))
        paths.append(str(path))
    review_set = model.resolve_inputs(paths)
    identities = [model.identify(p, 16, 12) for p in paths]
    record_path = str(tmp_path / "review.json")
    record = model.ReviewRecord.open(record_path, review_set, identities)
    record.set_rating(record.set_ids[0], 4)
    record.set_reviewer("Name")
    record.save()

    out = tmp_path / "share.md"
    assert tool.main([*paths, "--review", record_path, "--export", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "| r0.png | 4/5 |" in text and "Name" not in text and str(tmp_path) not in text
    # Without images the record is exported as it is.
    assert tool.main(["--review", record_path, "--export", str(tmp_path / "s.json")]) == 0

    cv2.imwrite(paths[1], np.zeros((12, 16, 3), np.uint8))
    assert tool.main([*paths, "--review", record_path, "--export", str(out)]) == 2
    assert "r1.png has changed" in capsys.readouterr().err
    assert tool.main([str(tmp_path / "absent.png")]) == 2
    assert "not found" in capsys.readouterr().err
