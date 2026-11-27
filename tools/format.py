#!/usr/bin/env python3
"""Formats the C, C++ and CUDA sources with clang-format as .clang-format describes, at the version requirements.txt
pins (another version lays some lines out differently).

    python tools/format.py                    every tracked source, in place (a file is written only if it changes)
    python tools/format.py FILE ...           these files
    python tools/format.py --check [--diff]   nothing written: the files it would change, each with its changes with
                                              --diff; exit 1 if there are any (CI)

clang-format: --clang-format PATH, else $CLANG_FORMAT, else the one pip installed beside this Python
(pip install -r requirements.txt), else PATH's.  The folders whose own .clang-format turns formatting off (vendored
code: third_party/, src/server/vendor/, tools/image/sdcpp_native/) stay as they are.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".inl")


def pinned_version() -> str:
    m = re.search(r"^clang-format==([0-9.]+)", (ROOT / "requirements.txt").read_text(encoding="utf-8"), re.M)
    if not m:
        sys.exit("format: requirements.txt pins no clang-format")
    return m.group(1)


def find_clang_format(given: str | None) -> str:
    if given:
        return given
    if os.environ.get("CLANG_FORMAT"):
        return os.environ["CLANG_FORMAT"]
    exe = "clang-format.exe" if os.name == "nt" else "clang-format"
    for d in (Path(sys.executable).parent, Path(sysconfig.get_path("scripts"))):
        if (d / exe).is_file():
            return str(d / exe)
    found = shutil.which("clang-format")
    if not found:
        sys.exit("format: no clang-format (pip install -r requirements.txt, or --clang-format PATH)")
    return found


def tracked_sources() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"format: the tracked files need git: {e}")
    names = [n.decode("utf-8", "surrogateescape") for n in out.split(b"\0") if n]
    return sorted(n for n in names if n.endswith(SUFFIXES))


def formatted(exe: str, path: Path) -> tuple[bytes, bytes]:
    """the file and clang-format's layout of it"""
    before = path.read_bytes()
    r = subprocess.run([exe, "--style=file", str(path)], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{path}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return before, r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="these files (default: every tracked source)")
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if a file would change")
    ap.add_argument("--diff", action="store_true", help="with --check, each file's changes")
    ap.add_argument("--clang-format", dest="exe", help="the clang-format to run")
    ap.add_argument("--any-version", action="store_true", help="run a clang-format other than the pinned version")
    a = ap.parse_args()

    exe = find_clang_format(a.exe)
    want = pinned_version()
    try:
        have = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"format: {exe} does not run: {e}")
    if f"version {want}" not in have and not a.any_version:
        sys.exit(f"format: {exe} is {have!r}, requirements.txt pins {want} (pip install -r requirements.txt, or "
                 "--any-version)")

    paths = [Path(f).resolve() for f in a.files] if a.files else [ROOT / n for n in tracked_sources()]
    with cf.ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        jobs = {pool.submit(formatted, exe, p): p for p in paths}
        results = {}
        for job in cf.as_completed(jobs):
            try:
                results[jobs[job]] = job.result()
            except (OSError, RuntimeError) as e:
                print(f"format: {e}", file=sys.stderr)
                return 2

    changed = [p for p in paths if results[p][0] != results[p][1]]
    for p in changed:
        name = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
        before, after = results[p]
        if a.check:
            print(name)
            if a.diff:
                sys.stdout.writelines(difflib.unified_diff(
                    before.decode("utf-8", "replace").splitlines(keepends=True),
                    after.decode("utf-8", "replace").splitlines(keepends=True), f"a/{name}", f"b/{name}"))
        else:
            p.write_bytes(after)
            print(f"formatted {name}")
    if a.check and changed:
        print(f"{len(changed)} of {len(paths)} files are not formatted: python tools/format.py formats them",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
