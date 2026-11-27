#!/bin/sh
# Lamina's draft layer for Qwen3.8-Flash-Next, one file of 0.8 GB, from
# https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP at a pinned revision, checked by its SHA-256.
#
#   ./download-draft-layer.sh FOLDER
#
# lamina takes the file from beside the model or beside the program; elsewhere name it in the config's "args":
# "--drafter", "FOLDER/Lamina-Qwen3.8-Flash-Next-MTP.gguf".  A download that broke off goes on from where it stopped.
# HF_ENDPOINT names a mirror.  It needs curl or wget, and sha256sum.
set -eu
name=Lamina-Qwen3.8-Flash-Next-MTP.gguf
if [ $# -ne 1 ]; then
  echo "usage: $0 FOLDER" >&2
  echo "Downloads Lamina's draft layer for Qwen3.8-Flash-Next ($name, 0.8 GB) into FOLDER: the model's" >&2
  echo "folder, where lamina finds it" >&2
  exit 2
fi
rev=e0d95e4236a43f3e02153c290211b74ba17a6f90
sha=eeaa94d4cd1e51f5b548aacbb72f8a0f6989cc11c62923b5cece11b33712c7db
url="${HF_ENDPOINT:-https://huggingface.co}/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP/resolve/$rev/$name"
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -L --fail --retry 5 --retry-delay 5 -C - -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -c -O "$2" "$1"; }
else
  echo "curl or wget is needed" >&2
  exit 1
fi
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum is needed (coreutils)" >&2; exit 1; }
sha_of() { sha256sum "$1" | cut -d ' ' -f 1; }
mkdir -p "$1"
file=$(cd "$1" && pwd)/$name

if [ -f "$file" ]; then
  if [ "$(sha_of "$file")" != "$sha" ]; then
    echo "$file is not the file expected (its SHA-256 differs): move it away and run this again" >&2
    exit 1
  fi
  echo "$name: there already, checked"
else
  if [ ! -f "$file.part" ] || [ "$(sha_of "$file.part")" != "$sha" ]; then
    echo "$name ..."
    if ! fetch "$url" "$file.part"; then
      echo "the download failed. Run this again to go on from where it stopped; if it fails the same way," \
           "delete $file.part first" >&2
      exit 1
    fi
    if [ "$(sha_of "$file.part")" != "$sha" ]; then
      echo "the download is not the file expected (its SHA-256 differs): delete $file.part and run this again" >&2
      exit 1
    fi
  fi
  mv "$file.part" "$file"
  echo "$name: downloaded, checked"
fi
echo
echo "The draft layer is $file"
echo "lamina takes it from beside the model or the program; elsewhere name it in the config's \"args\":"
echo "  \"--drafter\", \"$file\""
