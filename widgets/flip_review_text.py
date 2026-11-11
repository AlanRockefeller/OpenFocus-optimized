"""User-visible English strings of the flip review viewer.

Kept in one table, keyed like ``locales`` entries and formatted with
``str.format`` placeholders, so they can move into the translation tables
when the viewer is opened from the application.
"""

TEXT = {
    "window_title": "Flip review - {title}",
    "untitled": "untitled set",
    "loading": "Loading {done} / {total}...",
    "loading_full": "Loading full resolution...",
    "load_failed": "Could not open {name}: {error}",
    "no_images": "No images",
    "position": "{index} / {total}",
    "zoom_fit": "Fit {percent} %",
    "zoom_percent": "{percent} %",
    "best_marker": "best",
    "blind_letter": "Candidate {letter}",
    "diff_badge": "|difference| x{gain} vs {reference}",
    "diff_same": "Difference: this is the reference",
    "diff_computing": "Computing difference...",
    "pin_mode": "Click the image to place a note (Esc cancels)",
    "help_overlay": (
        "Left / Right  previous / next candidate\n"
        "Space  back to the previously viewed candidate\n"
        "Ctrl+1..9  show candidate k\n"
        "1..5  rate (press again to clear)    B  best of set\n"
        "+ / -  zoom    0  fit    Z  fit / 100 %    wheel: zoom at cursor\n"
        "drag  pan    F or F11  fullscreen    Esc  leave fullscreen\n"
        "H  hide labels    D  difference    Shift+D  difference gain\n"
        "R  use as difference reference    P or Ctrl+click  pin a note\n"
        "N  notes panel    C  comment    ? or F1  this help"),

    # panel
    "panel_candidates": "Candidates",
    "panel_current": "Current candidate",
    "rating": "Rating",
    "rating_clear": "Clear",
    "rating_tip": "Overall rating 1-5 (keys 1-5; press the same key again to clear)",
    "best_checkbox": "Best of this set",
    "best_tip": "One candidate per set can be marked best (key B)",
    "comment_placeholder": "Comment on this candidate",
    "criteria_toggle": "Criteria",
    "criteria_tip": "Optional ratings per criterion; leave empty if not needed",
    "criterion_none": "-",
    "criterion_sharpness": "Sharpness",
    "criterion_halos": "Halos",
    "criterion_seams": "Seams or artifacts",
    "criterion_noise": "Noise",
    "criterion_colour": "Colour",
    "pins": "Pinned notes",
    "pin_add": "Pin a note",
    "pin_add_tip": "Click a point of the image to attach a note there (P or Ctrl+click)",
    "pin_remove": "Remove",
    "pin_placeholder": "Note for the selected pin",
    "pin_item": "({x}, {y}) {candidate}: {text}",
    "pin_set": "set",
    "set_comment": "Set comment",
    "set_comment_placeholder": "Comment on the set as a whole",
    "reviewer": "Reviewer",
    "reviewer_placeholder": "Optional name (not exported unless chosen)",
    "blind": "Blind review",
    "blind_tip": "Replace labels by letters in a shuffled order until revealed",
    "reveal": "Reveal",
    "reveal_confirm": "Click again to reveal",
    "revealed": "Revealed",
    "export_group": "Share",
    "export": "Export review...",
    "export_tip": "Write scores and notes for sharing: no pixels, no file paths or names",
    "export_paths": "Include file paths",
    "export_reviewer": "Include reviewer name",
    "export_title": "Export review",
    "export_filter": "Review JSON (*.json);;Markdown (*.md)",
    "export_done": "Exported to {path}",
    "export_failed": "Export failed: {error}",
    "saved": "Saved {time}",
    "save_failed": "Could not save the review: {error}",
    "not_saving": "Not saving: the review record does not match these images",
    "mismatch": "{path} does not match these images:\n{problems}",
    "new_record": "Start a new record (keep the old one as a backup)",
    "record_path": "Record: {path}",
    "reference_tag": "reference",
}


def text(key: str, **values) -> str:
    """Look up and format one string; unknown keys show the key itself."""
    template = TEXT.get(key, key)
    return template.format(**values) if values else template
