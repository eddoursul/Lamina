#!/usr/bin/env python3
"""Lamina's CI (.github/workflows/build.yml), runnable on a PC the same way: the CUDA toolkit from NVIDIA's
redistributable archives, the portable preset built by build.bat or build.sh with the tests, the tests that need no
GPU, and the release archives from cmake --install.

    python .github/ci.py toolkit --dest DIR     the toolkit for this OS and CPU into DIR (in GitHub Actions also
                                                 CUDA_PATH and PATH for the later steps; elsewhere CUDACXX names it)
    python .github/ci.py build                  build-portable/: every program and test, and the image encoder (the
                                                 portable preset: the release's GPUs, for other PCs)
    python .github/ci.py test                   the tests that need no GPU: ctest -L cpu, the tools' unit tests
    python .github/ci.py package [--version V]  dist/lamina-V-bin-<os>-cuda-<x.y>-<cpu>.zip (.tar.gz on Linux), flat:
                                                 the server with the CUDA runtime linked in, the script that
                                                 downloads Flash-Next's draft layer and the licenses, NVIDIA's among
                                                 them (cmake --install's component lamina). On Linux the server's
                                                 libraries are checked first (GLIBC_MAX, SYSTEM_LIBS)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
ARM = platform.machine().lower() in ("arm64", "aarch64")
BUILD = ROOT / "build-portable"   # the portable preset's (CMakePresets.json)
# NVIDIA's redistributable CUDA 13.4.1 (redistrib_13.4.1.json): nvcc 13.4.59, the toolkit the PC Lamina is developed
# on builds with.  The parts the engine's build takes, nothing else.
CUDA_REDIST = "13.4.1"
REDIST_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
COMPONENTS = ("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cccl")
# What each archive holds (cmake --install's components): checked before it is written.  The server alone: the
# engine runs in its process (the engine alone, lamina-engine, is for tests and measurements)
EXECUTABLES = [f"lamina{EXE}"]
PROGRAMS = [*EXECUTABLES, "download-draft-layer.bat" if WIN else "download-draft-layer.sh", "LICENSE",
            "LICENSE-llama.cpp", "LICENSE-cpp-httplib", "LICENSE-jsonhpp", "LICENSE-Outfit", "LICENSE-cuda"]
# What the Linux programs may take from the system (docs/SETUP.md, Release binaries): glibc 2.34 at most (Ubuntu
# 22.04, Debian 12, RHEL 9, Amazon Linux 2023), OpenSSL 3; the C++ runtime (the portable preset) and the CUDA runtime
# are linked in, and the driver's libcuda is loaded by name at run time
GLIBC_MAX = (2, 34)
SYSTEM_LIBS = {"libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1", "ld-linux-x86-64.so.2",
               "ld-linux-aarch64.so.1", "libssl.so.3", "libcrypto.so.3"}


def say(msg=""):
    print(msg, flush=True)


def fail(msg):
    say(f"\nci stopped: {msg}")
    sys.exit(1)


def env() -> dict:
    """This environment without a PYTHONPATH or PIP_TARGET meant for another Python, with this Python's scripts
    folder first on the PATH (CMake and Ninja from pip, as in the workflow, in a virtual environment not activated)."""
    e = dict(os.environ)
    for k in ("PYTHONPATH", "PIP_TARGET"):
        e.pop(k, None)
    here = Path(sys.executable).parent
    e["PATH"] = os.pathsep.join([str(d) for d in ([here, here / "Scripts"] if WIN else [here])] + [e.get("PATH", "")])
    return e


def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    say("  $ " + " ".join(cmd))
    if subprocess.run(cmd, **kw).returncode != 0:
        fail(f"{Path(cmd[0]).name} failed (above)")


def tool(name: str) -> str:
    found = shutil.which(name, path=env()["PATH"])
    if not found:
        fail(f"{name} is not on the PATH")
    return found


def export(name: str, value: str):
    """`name` for this process and, in GitHub Actions, for the job's later steps."""
    os.environ[name] = value
    if os.environ.get("GITHUB_ENV"):
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def download(url: str, dst: Path, size: int, sha256: str):
    """url into dst, checked by its size and SHA-256; three tries."""
    for attempt in range(3):
        h, done = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
                while b := r.read(8 << 20):
                    f.write(b)
                    h.update(b)
                    done += len(b)
            if done == size and h.hexdigest() == sha256:
                return
            why = f"{done} bytes, SHA-256 {h.hexdigest()}"
        except OSError as e:
            why = str(e)
        dst.unlink(missing_ok=True)
        if attempt == 2:
            fail(f"{url}: {why}, not the archive NVIDIA's manifest names")
        say(f"  {dst.name}: {why}; trying again")
        time.sleep(10)


def redist_platform() -> str:
    if WIN:
        return "windows-arm64" if ARM else "windows-x86_64"
    return "linux-sbsa" if ARM else "linux-x86_64"


def extract(arc: Path, dest: Path):
    """An archive's files into dest, without its top folder (<component>-<platform>-<version>-archive)."""
    if arc.name.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            for m in z.infolist():
                rel = m.filename.split("/", 1)[1] if "/" in m.filename else ""
                if not rel or m.is_dir():
                    continue
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)
        return
    with tarfile.open(arc) as t:
        for m in t.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel:
                continue
            m.name = rel
            if m.islnk():
                m.linkname = m.linkname.split("/", 1)[1]
            t.extract(m, dest, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def toolkit(dest: Path, version: str):
    plat = redist_platform()
    mark, key = dest / ".lamina-redist", f"{version} {plat} {' '.join(COMPONENTS)}"
    if not (mark.exists() and mark.read_text().strip() == key):
        with urllib.request.urlopen(f"{REDIST_URL}/redistrib_{version}.json", timeout=60) as r:
            manifest = json.loads(r.read())
        dest.mkdir(parents=True, exist_ok=True)
        for c in COMPONENTS:
            e = manifest[c][plat]
            arc = dest.parent / Path(e["relative_path"]).name
            say(f"  {c} {manifest[c]['version']} ({int(e['size']) / 1e6:.0f} MB)")
            download(f"{REDIST_URL}/{e['relative_path']}", arc, int(e["size"]), e["sha256"])
            extract(arc, dest)
            arc.unlink()
        if not WIN and not (dest / "lib64").exists():   # nvcc.profile links from lib64, the archives hold lib
            (dest / "lib64").symlink_to("lib")
        mark.write_text(key + "\n")
    say(f"  ok: CUDA {version} ({plat}) in {dest}")
    export("CUDA_PATH", str(dest))
    if os.environ.get("GITHUB_PATH"):
        with open(os.environ["GITHUB_PATH"], "a", encoding="utf-8") as f:
            f.write(f"{dest / 'bin'}\n")
    else:
        say(f"  for the next steps: CUDACXX={dest / 'bin' / ('nvcc' + EXE)}")


def engine_version() -> str:
    """The version the programs report (CMakeLists.txt's project VERSION, LAMINA_VERSION)."""
    m = re.search(r"^project\(lamina VERSION ([0-9.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig"),
                  re.MULTILINE)
    return m.group(1) if m else "?"


def build():
    tag = os.environ.get("GITHUB_REF_NAME", "") if os.environ.get("GITHUB_REF_TYPE") == "tag" else ""
    if tag and tag != f"v{engine_version()}":   # before an hour of building
        fail(f"the tag {tag} is not the version the programs report, v{engine_version()}: set project(lamina "
             "VERSION ...) in CMakeLists.txt to the release's and tag that commit")
    e = env()
    ccache = shutil.which("ccache")
    if ccache:   # read at a build folder's first configure; the image encoder's build takes them from it
        for lang in ("C", "CXX", "CUDA"):
            e[f"CMAKE_{lang}_COMPILER_LAUNCHER"] = ccache
    script = [ROOT / "build.bat"] if WIN else ["sh", ROOT / "build.sh"]
    # LAMINA_JOBS, else on a runner, which does nothing else, every core (a PC's build leaves half to the desktop)
    jobs = os.environ.get("LAMINA_JOBS") or (str(os.cpu_count() or 4) if os.environ.get("GITHUB_ACTIONS") else "")
    jobs = [f"-DLAMINA_JOBS={jobs}"] if jobs else []
    run([*script, "portable", "-DLAMINA_BUILD_TESTS=ON", *jobs], env=e, cwd=ROOT)
    if ccache:
        subprocess.run([ccache, "--show-stats"])


def test():
    run([tool("ctest"), "--test-dir", BUILD, "-L", "cpu", "--output-on-failure", "--timeout", "900"], env=env())
    # the tools' tests (requirements.txt), from the checkout's folder as CONTRIBUTING.md runs them
    run([sys.executable, "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py"], cwd=ROOT, env=env())


def write_archive(base: Path, folder: Path) -> Path:
    """folder's files into base.zip on Windows, base.tar.gz elsewhere (programs executable), flat."""
    files = sorted(p for p in folder.iterdir() if p.is_file())
    if WIN:
        out = base.with_name(base.name + ".zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for src in files:
                z.write(src, src.name)
    else:
        out = base.with_name(base.name + ".tar.gz")
        with tarfile.open(out, "w:gz") as t:
            for src in files:
                info = t.gettarinfo(str(src), src.name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(src, "rb") as f:
                    t.addfile(info, f)
    say(f"  ok: {out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.0f} MB, {len(files)} files")
    return out


def readelf(*args: str) -> str:
    r = subprocess.run([tool("readelf"), "-W", *args], capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"readelf {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def check_linux_floor(folder: Path):
    """The programs in `folder` against SYSTEM_LIBS and GLIBC_MAX, with no RUNPATH (nothing loaded from their folder or
    the build machine's), and the C++ runtime they link in unexported (a system library's libstdc++ would bind to
    it), as readelf reads them."""
    problems, libs, newest = [], set(), (0, 0)
    for name in EXECUTABLES:
        exe = str(folder / name)
        # the runtime's own definitions (GLOBAL, UNIQUE); a template the program instantiates is WEAK, exported where
        # a library calls it (C++20 code instantiates std::string itself), as in any C++ program
        leaked = []
        for line in readelf("--dyn-syms", exe).splitlines():   # Num: Value Size Type Bind Vis Ndx Name
            f = line.split()
            if (len(f) >= 8 and f[6] != "UND" and f[4] != "WEAK"
                    and re.match(r"_ZN?K?St|__cxa_|__gxx_|_Unwind_", f[7])):
                leaked.append(f[7])
        if leaked:
            problems.append(f"{name} exports {len(leaked)} symbols of the C++ runtime it links in ({leaked[0]}, ...)")
        dynamic = readelf("-d", exe)
        for lib in re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", dynamic):
            libs.add(lib)
            if lib not in SYSTEM_LIBS:
                problems.append(f"{name} needs {lib}")
        paths = re.findall(r"\((?:RUNPATH|RPATH)\)\s+Library r(?:un)?path: \[(.*?)\]", dynamic)
        if paths:
            problems.append(f"{name} has a RUNPATH, {', '.join(paths)}")
        lib = "?"
        for line in readelf("-V", exe).splitlines():   # the version needs: "File: libc.so.6", "Name: GLIBC_2.34"
            if m := re.search(r"File: (\S+)", line):
                lib = m.group(1)
            elif m := re.search(r"Name: GLIBC_(\d+)\.(\d+)", line):
                v = (int(m.group(1)), int(m.group(2)))
                newest = max(newest, v)
                if v > GLIBC_MAX:
                    problems.append(f"{name} needs GLIBC_{v[0]}.{v[1]} ({lib})")
    if problems:
        fail("the Linux programs need more than the systems they are for (GLIBC_MAX, SYSTEM_LIBS): "
             + "; ".join(sorted(set(problems))))
    say(f"  ok: the programs need glibc {newest[0]}.{newest[1]} and " + ", ".join(sorted(libs)))


def package(version: str):
    cache = (BUILD / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    nvcc = re.search(r"^CMAKE_CUDA_COMPILER:\w+=(.+)$", cache, re.MULTILINE)
    v = re.search(r"release (\d+)\.(\d+)", subprocess.run([nvcc.group(1).strip(), "--version"], capture_output=True,
                                                          text=True).stdout) if nvcc else None
    if not v:
        fail(f"{BUILD} names no CUDA compiler that reports its version")
    major = v.group(1)
    name = f"{'win' if WIN else 'linux'}-cuda-{major}.{v.group(2)}-{'arm64' if ARM else 'x64'}"
    stage = BUILD / "package"
    shutil.rmtree(stage, ignore_errors=True)
    run([tool("cmake"), "--install", BUILD, "--component", "lamina", "--prefix", stage])
    missing = [n for n in PROGRAMS if not (stage / n).is_file()]
    if missing:
        fail("cmake --install's component lamina lacks " + ", ".join(missing))
    if not WIN:
        check_linux_floor(stage)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    write_archive(dist / f"lamina-{version}-bin-{name}", stage)


def default_version() -> str:
    """A tag's name, else the commit's (8 digits)."""
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        return os.environ["GITHUB_REF_NAME"]
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "local"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toolkit")
    t.add_argument("--dest", required=True)
    t.add_argument("--version", default=CUDA_REDIST, help=f"NVIDIA's redistrib_<version>.json (default {CUDA_REDIST})")
    sub.add_parser("build")
    sub.add_parser("test")
    p = sub.add_parser("package")
    p.add_argument("--version", help="the archives' version (default: the tag, else the commit)")
    a = ap.parse_args()
    if a.cmd == "toolkit":
        toolkit(Path(a.dest).resolve(), a.version)
    elif a.cmd == "build":
        build()
    elif a.cmd == "test":
        test()
    else:
        package(a.version or default_version())
    return 0


if __name__ == "__main__":
    sys.exit(main())
