"""lamina's settings table (src/server/params.cpp) against the config keys its sources read: every key the server
reads is an option of the table (or the config's "args" and "mcpServers"), and every key of the table is read
somewhere, so the config file and the command line stay one set of settings.

    python -m unittest tools.test_server_settings
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "src" / "server"
# a row of the table: {"--flag", "aliases", "key", K::Kind, ...
ROW = re.compile(r'\{"(--[a-z0-9-]+)",\s*"([^"]*)",\s*"([a-z_.]+)",\s*K::(\w+)')
# a key read from a config object: jget(cfg, "k"), jstr(cfg, "k"), cfg.contains("k"), cfg["k"], with the object's name
READ = re.compile(r'(?:jget|jstr)\(\s*([a-z_]+(?:\["[a-z_]+"\])?)\s*,\s*"([a-zA-Z_]+)"|'
                  r'\b([a-z_]+)\.contains\("([a-zA-Z_]+)"\)|\b([a-z_]+)\["([a-zA-Z_]+)"\]')
# the config objects of each file: the variable, and the section it holds ("": the top level)
OBJECTS = {
    "main.cpp": {"cfg": "", 'cfg["vision"]': "vision", "given": "", "file_cfg": ""},
    "config.cpp": {"cfg": "", 'cfg["vision"]': "vision"},
    "service.cpp": {"cfg": ""},
    "mcp.cpp": {"cfg": ""},
    "model_config.cpp": {"c": "", "cfg": ""},
    "vision.cpp": {"cfg": "vision"},
    "images.cpp": {"cfg": "images"},
}
# a file whose other functions call other objects "cfg" (mcp.cpp's server entries): only these functions are read
FUNCTIONS = {"mcp.cpp": ("settings_from", "hub_from_config")}
# keys read through a list of names: for (const char* k : {"exe", ...}) jget(cfg, k)
LOOPED = re.compile(r'for \(const char\* k : \{([^}]*)\}\)\s*\{?\s*(?:const json& v = )?jget\(cfg, k\)')
# keys compared in a function of their section: settings_from (mcp), sampling_defaults_from_config (sampling)
COMPARED = {"mcp.cpp": ("settings_from", "mcp"), "service.cpp": ("sampling_defaults_from_config", "sampling")}
# read, and not settings: the config's engine options, the other spelling of mcp_servers, the sampling keys' other
# spellings (repeat_penalty, repeat_last_n: merged into the table's), and replicas, which a config is refused for
NOT_SETTINGS = {"args", "mcpServers", "sampling.repeat_penalty", "sampling.repeat_last_n", "replicas"}
# options taken and not read: a config or a command line of an earlier version may hold them
NOT_READ = {"vision.model", "lib_dirs"}


def table() -> dict[str, tuple[str, str, str]]:
    text = (SERVER / "params.cpp").read_text(encoding="utf-8")
    return {key: (flag, aliases, kind) for flag, aliases, key, kind in ROW.findall(text)}


def keys_read() -> set[str]:
    found: set[str] = set()
    for name, objects in OBJECTS.items():
        text = (SERVER / name).read_text(encoding="utf-8")
        if name in FUNCTIONS:
            text = "".join(body(text, f) for f in FUNCTIONS[name])
        for m in LOOPED.finditer(text):
            section = objects["cfg"]
            found |= {f"{section}.{k}" if section else k for k in re.findall(r'"([a-z_]+)"', m.group(1))}
        for m in READ.finditer(text):
            obj, key = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4)) if m.group(3) else \
                (m.group(5), m.group(6))
            if obj in objects:
                section = objects[obj]
                found.add(f"{section}.{key}" if section else key)
    for name, (func, section) in COMPARED.items():
        text = body((SERVER / name).read_text(encoding="utf-8"), func)
        found |= {f"{section}.{k}" for k in re.findall(r'key == "([a-z_]+)"', text)}
    return found


def body(text: str, func: str) -> str:
    """a function's definition: from its name to the brace that ends it at the line's start"""
    at = re.search(r"\n[^\n]*\b" + func + r"\(", text).start()
    return text[at:text.index("\n}\n", at)]


class SettingsTable(unittest.TestCase):
    def test_every_key_read_is_an_option(self):
        keys = set(table())
        tops = {k.split(".")[0] for k in keys}
        missing = sorted(k for k in keys_read() - NOT_SETTINGS if k not in keys and k not in tops)
        self.assertEqual(missing, [], "config keys the server reads that src/server/params.cpp has no option for")

    def test_every_option_is_read(self):
        read = keys_read()
        sections = {k.split(".")[0] for k in read}
        unread = sorted(k for k in table() if k not in read and k not in sections and k not in NOT_READ)
        self.assertEqual(unread, [], "options of src/server/params.cpp whose key the server does not read")

    def test_options_named_after_their_keys(self):
        # an option is its key with "-" for "_" ("." too, after a section's name), but for the sampling defaults
        # (llama.cpp's names) and the spellings kept from before
        kept = {"model": "--model", "vision": "--no-mmproj", "vision.mmproj": "--mmproj", "mcp_config": "--mcp-config",
                "scripts": "--script", "log": "--log", "lazy_load": "--lazy-load", "idle_unload_s": "--idle-unload-s",
                "open_browser": "--open-browser"}
        for key, (flag, _aliases, _kind) in table().items():
            if key.startswith("sampling.") or key in kept:
                continue
            self.assertEqual(flag, "--" + key.replace("_", "-").replace(".", "-"), key)


if __name__ == "__main__":
    unittest.main()
