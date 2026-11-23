"""Each locale table lists every key once.

Python keeps the last value of a repeated key in a dict literal without any
warning, so a repeated key hides one translation. The key-set parity tests
compare the resulting dicts and cannot see this.
"""
import ast
from pathlib import Path

import pytest

LOCALES = sorted((Path(__file__).resolve().parent.parent / 'locales').glob('*.py'))


def _repeated_keys(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    repeated = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        first = {}
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value in first:
                    repeated.append((key.value, first[key.value], key.lineno))
                else:
                    first[key.value] = key.lineno
    return repeated


def test_locale_files_are_found():
    assert {'en.py', 'de.py', '_extra.py'} <= {path.name for path in LOCALES}


@pytest.mark.parametrize('path', LOCALES, ids=lambda path: path.name)
def test_locale_dicts_have_no_repeated_keys(path):
    assert _repeated_keys(path) == []
