"""Tests that the source distribution carries the project and nothing else.

There is no ``MANIFEST.in``: setuptools-scm's file finder builds the sdist
from the tracked file list, so whatever is committed is what ships. That keeps
the packaging honest as long as the repository stays tidy --- a scratch config
or a stray ``*.bak`` dropped at the root would be swept straight into the next
release. The repo keeps such things under ``backup/``, which is gitignored.

So the guard is on what is tracked: every file belongs under one of a handful
of top-level directories or is one of a named set of root files. Anything else
fails here, on the commit that added it, rather than in a release tarball
nobody inspects.

Test cases:
- test_tracked_files_live_in_known_top_level_directories: no stray directories
- test_only_the_expected_files_are_tracked_at_the_repo_root: no stray root files
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]

# Top-level directories whose contents are all legitimately part of the
# project and, through the sdist, of a release.
ALLOWED_DIRS = {
    "src",
    "tests",
    "doc",
    "examples",
    "tutorial",
    "bin",
    ".github",
}

# Files allowed to sit directly at the repo root.
ALLOWED_ROOT_FILES = {
    "COPYING",
    "README.md",
    "pyproject.toml",
    "pytest.ini",
    "Makefile",
    ".gitignore",
    ".pre-commit-config.yaml",
}

_HINT = "move scratch files under backup/ (gitignored) or add them to an allowed location"


def _tracked_files():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(p) for p in out.split("\0") if p]


def test_tracked_files_live_in_known_top_level_directories():
    stray = sorted(
        {
            str(path)
            for path in _tracked_files()
            if len(path.parts) > 1 and path.parts[0] not in ALLOWED_DIRS
        }
    )
    assert not stray, f"tracked files in an unexpected directory ({_HINT}): {stray}"


def test_only_the_expected_files_are_tracked_at_the_repo_root():
    stray = sorted(
        path.name
        for path in _tracked_files()
        if len(path.parts) == 1 and path.name not in ALLOWED_ROOT_FILES
    )
    assert not stray, f"unexpected file tracked at the repo root ({_HINT}): {stray}"
