#!/usr/bin/env python3
"""The sources' checks, each tool at the version requirements-lint.txt pins: the C, C++ and CUDA sources' layout
(clang-format: the files tools/format.py would change), Python's errors (ruff, as ruff.toml selects: undefined and
unused names, syntax errors; not its style), the shell scripts (shellcheck), the workflows (actionlint) and the
Markdown files' relative links (a file that exists, a heading GitHub makes the anchor from).

    python tools/lint.py            the tracked files as the working tree has them (CI's lint job)
    python tools/lint.py --staged   the staged files as the index has them: what the commit takes (.githooks/pre-commit)

Exit 1 with each problem named.  The tools: pip install -r requirements-lint.txt; each is the one beside this Python,
else the PATH's.  Vendored code (third_party/, src/server/vendor/, tools/image/sdcpp_native/) is not checked.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import os
import posixpath
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements-lint.txt"
VENDORED = ("third_party/", "src/server/vendor/", "tools/image/sdcpp_native/")
CPP = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".inl")
WORKFLOWS = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")
# each tool: its pip package and how it reports its version
TOOLS = {"clang-format": ("clang-format", "--version"), "ruff": ("ruff", "--version"),
         "shellcheck": ("shellcheck-py", "--version"), "actionlint": ("actionlint-py", "-version")}


def pinned(package: str) -> str:
    m = re.search(rf"^{re.escape(package)}==([0-9.]+)", REQUIREMENTS.read_text(encoding="utf-8"), re.M)
    if not m:
        sys.exit(f"lint: {REQUIREMENTS.name} pins no {package}")
    return m.group(1)


def find_tool(name: str) -> str | None:
    """the tool pip installed beside this Python, else the PATH's"""
    exe = name + (".exe" if os.name == "nt" else "")
    for d in (Path(sys.executable).parent, Path(sysconfig.get_path("scripts"))):
        if (d / exe).is_file():
            return str(d / exe)
    return shutil.which(name)


def tool(name: str, any_version: bool = False) -> str:
    """the tool's path, at the pinned version (a pip package's fourth number is the package's own)"""
    package, flag = TOOLS[name]
    exe = find_tool(name)
    if not exe:
        sys.exit(f"lint: no {name} (pip install -r {REQUIREMENTS.name})")
    try:
        have = subprocess.run([exe, flag], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"lint: {exe} does not run: {e}")
    want = ".".join(pinned(package).split(".")[:3])
    if not any_version and not re.search(rf"(?<![\d.]){re.escape(want)}(?![\d.])", have):
        first = have.strip().splitlines()[0] if have.strip() else "?"
        sys.exit(f"lint: {exe} is {first!r}, {REQUIREMENTS.name} pins {package}=={pinned(package)} "
                 f"(pip install -r {REQUIREMENTS.name})")
    return exe


def git(*args: str) -> bytes:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    except OSError as e:
        sys.exit(f"lint: git does not run: {e}")
    if r.returncode != 0:
        sys.exit(f"lint: git {' '.join(args)}: {text(r.stderr).strip() or r.returncode}")
    return r.stdout


def names(out: bytes) -> list[str]:
    return [n.decode("utf-8", "surrogateescape") for n in out.split(b"\0") if n]


class Tree:
    """the files checked: the working tree's tracked files, or the index's"""

    def __init__(self, staged: bool):
        self.staged = staged
        self.paths = names(git("ls-files", "-z"))
        if not staged:
            self.paths = [p for p in self.paths if (ROOT / p).is_file()]   # a deletion not staged yet
        self.files = set(self.paths)
        self.dirs = {d for p in self.paths for d in parents(p)}
        self.cache: dict[str, bytes] = {}

    def read(self, path: str) -> bytes:
        if path not in self.cache:
            self.cache[path] = git("show", f":{path}") if self.staged else (ROOT / path).read_bytes()
        return self.cache[path]


def parents(path: str):
    d = posixpath.dirname(path)
    while d:
        yield d
        d = posixpath.dirname(d)


def piped(cmd: list[str], data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=data, cwd=ROOT, capture_output=True)


def text(b: bytes) -> str:
    return b.decode("utf-8", "replace")


def check_layout(exe: str, tree: Tree, path: str) -> list[str]:
    data = tree.read(path)
    r = piped([exe, "--style=file", f"--assume-filename={path}"], data)
    if r.returncode != 0:
        return [f"{path}: clang-format failed: {text(r.stderr).strip()}"]
    if r.stdout == data:
        return []
    diff = "".join(difflib.unified_diff(text(data).splitlines(keepends=True), text(r.stdout).splitlines(keepends=True),
                                        f"a/{path}", f"b/{path}"))
    return [f"{path}: not formatted (python tools/format.py {path})\n{diff}".rstrip("\n")]


def check_python(exe: str, tree: Tree, path: str) -> list[str]:
    r = piped([exe, "check", "--force-exclude", "--no-cache", "--output-format", "concise", "--quiet",
               "--stdin-filename", path, "-"], tree.read(path))
    if r.returncode == 0:
        return []
    out = (text(r.stdout) + text(r.stderr)).strip().replace(path.replace("/", os.sep), path)
    return out.splitlines() or [f"{path}: ruff failed ({r.returncode})"]


def check_shell(exe: str, tree: Tree, path: str) -> list[str]:
    r = piped([exe, "--severity=warning", "--format=gcc", "-"], tree.read(path))
    if r.returncode == 0:
        return []
    out = (text(r.stdout) + text(r.stderr)).strip()
    return [re.sub(r"^-:", f"{path}:", line) for line in out.splitlines()] or [f"{path}: shellcheck failed"]


def check_workflow(exe: str, shellcheck: str, tree: Tree, path: str) -> list[str]:
    r = piped([exe, "-no-color", "-oneline", f"-shellcheck={shellcheck}", "-pyflakes=", "-stdin-filename", path, "-"],
              tree.read(path))
    if r.returncode == 0:
        return []
    return (text(r.stdout) + text(r.stderr)).strip().splitlines() or [f"{path}: actionlint failed"]


def slug(heading: str) -> str:
    """the anchor GitHub makes from a heading"""
    h = re.sub(r"`|\*\*|\[|\]\([^)]*\)", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")


def anchors(md: str) -> set[str]:
    out, fence = set(), False
    for line in md.splitlines():
        if line.startswith("```"):
            fence = not fence
        elif not fence and line.startswith("#"):
            out.add(slug(line.lstrip("#")))
    return out


def check_links(tree: Tree, path: str) -> list[str]:
    md = re.sub(r"```.*?```", "", text(tree.read(path)), flags=re.S)
    problems = []
    for target in re.findall(r"\]\(([^)\s]+)\)", md):
        if re.match(r"[a-z][a-z0-9+.-]*:", target):   # https:, mailto:
            continue
        rel, _, anchor = target.partition("#")
        dest = posixpath.normpath(posixpath.join(posixpath.dirname(path), rel)) if rel else path
        if dest not in tree.files and dest not in tree.dirs:
            problems.append(f"{path}: {target}: no such file")
        elif anchor and dest.endswith(".md") and dest in tree.files and anchor not in anchors(text(tree.read(dest))):
            problems.append(f"{path}: {target}: no heading for #{anchor}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", action="store_true", help="the staged files as the index has them")
    ap.add_argument("--any-version", action="store_true", help="run tools other than the pinned versions")
    a = ap.parse_args()

    tree = Tree(a.staged)
    if a.staged:   # the files the commit adds or changes; every Markdown file if it adds, changes or takes one away
        changed = names(git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"))
        gone = names(git("diff", "--cached", "--name-only", "-z", "--diff-filter=DR"))
        check = [p for p in changed if not p.startswith(VENDORED)]
        if any(p.endswith(".md") for p in check) or gone:
            check += [p for p in tree.paths if p.endswith(".md") and p not in check]
    else:
        check = tree.paths
    check = [p for p in check if not p.startswith(VENDORED)]
    groups = {
        "clang-format": [p for p in check if p.endswith(CPP)],
        "ruff": [p for p in check if p.endswith(".py")],
        "shellcheck": [p for p in check if p.endswith(".sh")],
        "actionlint": [p for p in check if WORKFLOWS.match(p)],
    }
    md = [p for p in check if p.endswith(".md")]
    exes = {name: tool(name, a.any_version) for name, paths in groups.items()   # actionlint runs shellcheck too
            if paths or (name == "shellcheck" and groups["actionlint"])}
    jobs = []
    with cf.ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        for p in groups["clang-format"]:
            jobs.append(pool.submit(check_layout, exes["clang-format"], tree, p))
        for p in groups["ruff"]:
            jobs.append(pool.submit(check_python, exes["ruff"], tree, p))
        for p in groups["shellcheck"]:
            jobs.append(pool.submit(check_shell, exes["shellcheck"], tree, p))
        for p in groups["actionlint"]:
            jobs.append(pool.submit(check_workflow, exes["actionlint"], exes["shellcheck"], tree, p))
        for p in md:
            jobs.append(pool.submit(check_links, tree, p))
        problems = [line for job in jobs for line in job.result()]
    for line in problems:
        print(line)
    sys.stdout.flush()
    checked = ", ".join(f"{len(v)} {k}" for k, v in [*groups.items(), ("Markdown", md)] if v) or "nothing"
    if problems:
        print(f"lint: {len(problems)} problem{'s' * (len(problems) != 1)}; checked: {checked}", file=sys.stderr)
        return 1
    print(f"lint: ok; checked: {checked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
