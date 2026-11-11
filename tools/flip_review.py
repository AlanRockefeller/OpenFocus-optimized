"""Flip between renders of one focus stack and record scores and notes.

    python tools/flip_review.py before.tif after.tif [more.tif ...]
    python tools/flip_review.py results/                 # every image in a folder
    python tools/flip_review.py review-set.json          # a review-set manifest

The review record is saved automatically beside the images (or at
``--review PATH``). ``--export OUT.json|OUT.md`` writes a shareable summary
of an existing record without opening a window. See docs/FLIP_REVIEW.md.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import flip_review as model  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flip_review",
        description="Compare images of identical size in one shared view and record "
                    "ratings, comments and pinned notes.")
    parser.add_argument("inputs", nargs="*",
                        help="image files, one folder, or one review-set manifest (.json)")
    parser.add_argument("--review", metavar="PATH",
                        help="review record to use (default: beside the images)")
    parser.add_argument("--new-review", action="store_true",
                        help="if the record does not match these images, keep it as a "
                             "backup and start a new one")
    parser.add_argument("--title", help="title for a new review of files or a folder")
    parser.add_argument("--blind", action="store_true",
                        help="hide labels behind shuffled letters until revealed")
    parser.add_argument("--reviewer", metavar="NAME", help="reviewer name stored in the record")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--memory-mb", type=int, metavar="MB",
                        help="budget for full-resolution display buffers "
                             "(default: a quarter of available RAM, at most 2048)")
    parser.add_argument("--export", metavar="OUT",
                        help="write a shareable export (.json or .md) of the record and exit")
    parser.add_argument("--export-paths", action="store_true",
                        help="include candidate file paths in the export")
    parser.add_argument("--export-reviewer", action="store_true",
                        help="include the reviewer name in the export")
    return parser


def _identities(review_set: model.ReviewSet):
    """Decode headers via the application decoder for exact display sizes."""
    from core.image_loader import decode_image
    identities = []
    for candidate in review_set.candidates:
        image = decode_image(candidate.path, preserve_grayscale=True,
                             allow_transparent_png=True)
        if image is None:
            raise model.ReviewError(f"Could not decode {candidate.path}")
        height, width = image.shape[:2]
        del image
        identities.append(model.identify(candidate.path, width, height))
    return identities


def _export(args, review_set) -> int:
    path = args.review or (review_set.default_record_path() if review_set else None)
    if not path:
        print("error: --export needs --review PATH or the images of the set", file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print(f"error: no review record at {path}", file=sys.stderr)
        return 2
    record = model.ReviewRecord.read(path)
    if review_set is not None:
        record.bind(review_set, _identities(review_set))
    record.export(args.export, include_paths=args.export_paths,
                  include_reviewer=args.export_reviewer)
    print(f"exported {args.export}")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        review_set = model.resolve_inputs(args.inputs) if args.inputs else None
        if review_set is not None and args.title:
            review_set.title = args.title
        if args.export:
            return _export(args, review_set)
        if review_set is None:
            _parser().print_usage(sys.stderr)
            print("error: give images, a folder or a review-set manifest", file=sys.stderr)
            return 2
    except model.ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication

    from ui.styles import get_theme_style, set_current_theme
    from widgets.flip_review import FlipReviewViewer
    from widgets.flip_review_text import text

    app = QApplication.instance() or QApplication(sys.argv[:1])
    set_current_theme(args.theme)
    app.setStyleSheet(get_theme_style(args.theme))
    record_path = args.review or review_set.default_record_path()
    viewer = FlipReviewViewer(
        review_set, record_path,
        memory_budget=args.memory_mb * (1 << 20) if args.memory_mb else None,
        blind=args.blind, reviewer=args.reviewer)
    viewer.setWindowTitle(text("window_title", title=review_set.title or text("untitled")))
    exit_code = {"value": 0}

    def failed(message):
        print(f"error: {message}", file=sys.stderr)
        exit_code["value"] = 1

    def mismatch(message):
        if args.new_review:
            viewer.start_new_record()
            print(f"started a new review record at {record_path}", file=sys.stderr)
        else:
            print(f"warning: {message}\n(not saving; pass --new-review or --review PATH)",
                  file=sys.stderr)

    viewer.loadFailed.connect(failed)
    viewer.recordMismatch.connect(mismatch)
    viewer.saveFailed.connect(failed)
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        viewer.resize(int(area.width() * 0.85), int(area.height() * 0.85))
    if args.fullscreen:
        viewer.showFullScreen()
    else:
        viewer.show()
    viewer.focus_canvas()
    app.aboutToQuit.connect(viewer.shutdown)
    code = app.exec()
    viewer.shutdown()
    return code or exit_code["value"]


if __name__ == "__main__":
    sys.exit(main())
