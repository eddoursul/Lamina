// src/server/remote_model.cpp - the models a URL names, judged from their files' headers (remote_model.hpp).
#include "remote_model.hpp"

#include "http_client.hpp"
#include "json.hpp"
#include "model_config.hpp"
#include "process.hpp"

#include "lamina/artifact/draft_layer.hpp"
#include "lamina/artifact/drafter.hpp"
#include "lamina/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lamina {
namespace fs = std::filesystem;
namespace {

constexpr uint64_t kMiB = 1048576;
constexpr uint64_t kHeaderMax = 256 * kMiB;    // a GGUF header past this is refused (the largest known: ~20 MB)
constexpr uint64_t kJsonMax = 1 * kMiB;        // a JSON file read whole
constexpr uint64_t kSafetensorsMax = 64 * kMiB;   // as artifact/drafter.hpp's safetensors_header reads one
constexpr double kTimeout = 60.0;               // the connection and each read

/// %XX for each byte but the URL's unreserved ones (and '/' with `keep_slash`)
std::string encode(std::string_view s, bool keep_slash) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '.' || c == '_' || c == '~' || (keep_slash && c == '/')) {
            out += (char) c;
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

struct UrlParts {
    std::string scheme, host, path, query;   // host in lower case, without a user or a port; path from its '/'
    std::string port;                        // as given ("": the scheme's)
    std::string origin;                      // scheme://host[:port] as given
};

UrlParts split_url(const std::string& url) {
    UrlParts u;
    const size_t s = url.find("://");
    if (s == std::string::npos) return u;
    u.scheme = lower(url.substr(0, s));
    std::string rest = url.substr(s + 3);
    if (const size_t frag = rest.find('#'); frag != std::string::npos) rest.resize(frag);
    if (const size_t q = rest.find('?'); q != std::string::npos) {
        u.query = rest.substr(q + 1);
        rest.resize(q);
    }
    const size_t slash = rest.find('/');
    std::string authority = rest.substr(0, slash);
    u.path = slash == std::string::npos ? "" : rest.substr(slash);
    if (const size_t at = authority.rfind('@'); at != std::string::npos) authority = authority.substr(at + 1);
    u.origin = u.scheme + "://" + authority;
    std::string host = lower(authority);
    if (const size_t colon = host.rfind(':'); colon != std::string::npos && host.find(']') == std::string::npos) {
        u.port = host.substr(colon + 1);
        host.resize(colon);
    }
    u.host = host;
    return u;
}

/// a host as the Hub's are compared: without "www."
std::string bare_host(const std::string& h) { return starts_with(h, "www.") ? h.substr(4) : h; }

/// where a URL's requests go, as the token's one place is compared: scheme://host[:port], the host without "www.",
/// the scheme's own port left out
std::string origin_key(const std::string& url) {
    const UrlParts u = split_url(url);
    const bool own = u.port.empty() || (u.scheme == "https" && u.port == "443") || (u.scheme == "http" && u.port == "80");
    return u.scheme + "://" + bare_host(u.host) + (own ? "" : ":" + u.port);
}

/// where a redirect's "location" points, from the URL that gave it
std::string resolve_location(const std::string& from, const std::string& location) {
    if (is_url(location)) return location;
    const UrlParts u = split_url(from);
    if (starts_with(location, "//")) return u.scheme + ":" + location;
    if (starts_with(location, "/")) return u.origin + location;
    const size_t last = u.path.rfind('/');
    return u.origin + (last == std::string::npos ? "/" : u.path.substr(0, last + 1)) + location;
}

/// a path's folder ("" for one in the root) and its name
std::string parent_of(const std::string& p) {
    const size_t s = p.rfind('/');
    return s == std::string::npos ? "" : p.substr(0, s);
}
std::string name_of(const std::string& p) {
    const size_t s = p.rfind('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

/// gguf_split_paths' rule on a file name: "<stem>-NNNNN-of-MMMMM.gguf", its shard number and count (1 <= no <= count);
/// `at` where the first five digits start
bool split_name(const std::string& name, int& no, int& count, size_t& at) {
    static const char tag[] = "-of-";
    const size_t t = name.rfind(tag);
    if (t == std::string::npos || t < 6 || name[t - 6] != '-' || name.size() != t + 4 + 5 + 5 ||
        name.compare(t + 9, 5, ".gguf") != 0)
        return false;
    no = count = 0;
    for (size_t i = 0; i < 5; ++i) {
        const char a = name[t - 5 + i], b = name[t + 4 + i];
        if (a < '0' || a > '9' || b < '0' || b > '9') return false;
        no = no * 10 + (a - '0');
        count = count * 10 + (b - '0');
    }
    at = t - 5;
    return count >= 1 && no >= 1 && no <= count;
}
std::string with_shard(std::string name, size_t at, int no) {
    char num[12];
    std::snprintf(num, sizeof num, "%05d", no);
    name.replace(at, 5, num);
    return name;
}

/// the error text of a refusal: the status, what the Hub said, what it means (the caller names what was asked for)
std::string refusal(const HttpReply& r, bool token) {
    std::string said;
    json j;
    if (py_loads(r.body, j) && j.is_object()) said = jstr(j, "error");
    if (said.size() > 300) said = said.substr(0, 300) + "...";
    const std::string status = "HTTP " + std::to_string(r.status) + (said.empty() ? "" : ", \"" + said + "\"");
    if (r.status == 401)
        return "refused (" + status + ")" +
               (token ? ": the token (HF_TOKEN) cannot read it, or it is not valid"
                      : ": there is none, or a private or gated one, which needs a token (HF_TOKEN)");
    if (r.status == 403)
        return "refused (" + status + "): a gated repository needs its terms accepted on its page, and a token of "
               "that account (HF_TOKEN)";
    if (r.status == 404) return "not found (" + status + ")";
    return status + (r.reason.empty() ? "" : " " + r.reason);
}

/// A GET, its redirects followed here: the token goes to the Hub alone, `hub` its origin_key (a download's redirect
/// points at another host, with a signed URL)
HttpReply get(const std::string& url, const std::string& hub, const std::string& token, HttpHeaders extra,
              size_t max_bytes, const std::function<bool(std::string_view)>& on_data, const std::function<void()>& on_hop,
              std::string* final_url = nullptr) {
    std::string at = url;
    for (int hop = 0; hop < 10; ++hop) {
        HttpHeaders h = extra;
        h.emplace_back("User-Agent", "lamina");
        if (!token.empty() && !hub.empty() && origin_key(at) == hub)
            h.emplace_back("Authorization", "Bearer " + token);
        HttpReply r = http_request("GET", at, h, "", kTimeout, max_bytes, on_data, false);
        const std::string loc = r.header("location");
        if (r.status >= 300 && r.status < 400 && !loc.empty()) {
            at = resolve_location(at, loc);
            if (on_hop) on_hop();
            continue;
        }
        if (final_url != nullptr) *final_url = at;
        return r;
    }
    throw RemoteError(url + ": more than 10 redirects");
}

/// the Hub's JSON at `url`; `next` the Link header's rel="next" (a listing's next page), "" where there is none (`url`
/// a copy: a caller's loop passes its next page in both)
json hub_json(const std::string url, const std::string& host, const std::string& token, const std::string& what,
              std::string* next = nullptr) {
    HttpReply r;
    try {
        r = get(url, host, token, {{"Accept", "application/json"}}, 256 * kMiB, nullptr, nullptr);
    } catch (const HttpError& e) {
        throw RemoteError(what + ": " + e.what());
    }
    if (r.status != 200) throw RemoteError(what + ": " + refusal(r, !token.empty()));
    json j;
    std::string err;
    if (!py_loads(r.body, j, &err)) throw RemoteError(what + ": the Hub's answer is not JSON (" + err + ")");
    if (next != nullptr) {
        next->clear();
        // Link: <https://huggingface.co/api/...&cursor=...>; rel="next"
        for (const std::string& part : split(r.header("link"), ",")) {
            const size_t lt = part.find('<'), gt = part.find('>');
            if (lt != std::string::npos && gt != std::string::npos && gt > lt &&
                part.find("rel=\"next\"", gt) != std::string::npos)
                *next = resolve_location(url, part.substr(lt + 1, gt - lt - 1));
        }
    }
    return j;
}

RemoteListing list_hub(const HfTarget& t, const std::string& url, const std::string& token) {
    RemoteListing l;
    l.url = url;
    l.hub = true;
    l.hub_origin = origin_key(t.endpoint);
    l.repo = t.repo;
    const std::string kind = t.dataset ? "datasets" : "models";
    const std::string what = std::string(t.dataset ? "the dataset " : "") + t.repo;
    const std::string api = t.endpoint + "/api/" + kind + "/" + encode(t.repo, true);
    const json rev = hub_json(api + "/revision/" + encode(t.revision, false), l.hub_origin, token,
                              what + (t.revision == "main" ? "" : " at " + t.revision));
    l.commit = jstr(rev, "sha");
    if (l.commit.empty()) throw RemoteError(what + ": the Hub names no commit of " + t.revision);
    l.base = t.endpoint + "/" + (t.dataset ? "datasets/" : "") + encode(t.repo, true) + "/resolve/" + l.commit + "/";
    // the files of the folder the URL names (a file's: its folder), its subfolders' too
    auto list = [&](const std::string& dir) {
        l.files.clear();
        std::string next = api + "/tree/" + l.commit + (dir.empty() ? "" : "/" + encode(dir, true)) + "?recursive=true";
        while (!next.empty()) {
            const json page = hub_json(next, l.hub_origin, token, what + (dir.empty() ? "" : ", " + dir), &next);
            if (!page.is_array()) throw RemoteError(what + ": the Hub's listing is not a list");
            for (const json& e : page) {
                if (jstr(e, "type") != "file") continue;
                RemoteFile f;
                f.path = jstr(e, "path");
                f.url = l.base + encode(f.path, true);
                const json& lfs = jget(e, "lfs");
                const json& size = lfs.is_object() ? jget(lfs, "size") : jget(e, "size");
                f.size = size.is_number() ? size.get<uint64_t>() : 0;
                f.sha256 = lfs.is_object() ? jstr(lfs, "oid") : "";
                f.id = !f.sha256.empty() ? f.sha256 : !jstr(e, "oid").empty() ? "git:" + jstr(e, "oid") : f.url;
                l.files.push_back(std::move(f));
            }
        }
        std::sort(l.files.begin(), l.files.end(), [](const RemoteFile& a, const RemoteFile& b) { return a.path < b.path; });
    };
    bool file = t.file;
    if (!file) {
        try {
            list(t.path);
        } catch (const RemoteError&) {
            // tree/REVISION/FILE names a file the way blob/ does
            if (t.path.empty() || !ends_with(lower(t.path), ".gguf")) throw;
            file = true;
        }
    }
    if (file) {
        list(parent_of(t.path));
        if (l.find(t.path) == nullptr)
            throw RemoteError(what + ": no file " + t.path + " at " + (t.revision == "main" ? "main" : t.revision));
        l.focus = t.path;
        l.scope = parent_of(t.path);
    } else {
        l.scope = t.path;
    }
    return l;
}

RemoteListing list_direct(const std::string& url) {
    const UrlParts u = split_url(url);
    const size_t last = u.path.rfind('/');
    const std::string raw = last == std::string::npos ? u.path : u.path.substr(last + 1);
    const std::string folder = last == std::string::npos ? "/" : u.path.substr(0, last + 1);
    std::string name = unquote_to_bytes(raw);
    if (name.empty()) throw RemoteError(url + ": names no file");
    // a download host's URL may end in no .gguf (a hash, a query): the file is read as a GGUF all the same, its copy
    // named so (its header says whether it is one)
    if (!ends_with(lower(name), ".gguf")) name += ".gguf";
    RemoteListing l;
    l.url = url;
    l.base = u.origin + folder;
    l.focus = name;
    const std::string query = u.query.empty() ? "" : "?" + u.query;
    int no = 0, count = 0;
    size_t at = 0, raw_at = 0;
    int raw_no = 0, raw_count = 0;
    if (split_name(name, no, count, at) && split_name(raw, raw_no, raw_count, raw_at)) {
        for (int i = 1; i <= count; ++i) {
            RemoteFile f;
            f.path = with_shard(name, at, i);
            f.url = l.base + with_shard(raw, raw_at, i) + query;
            f.id = f.url;
            l.files.push_back(std::move(f));
        }
    } else {
        RemoteFile f;
        f.path = name;
        f.url = url;
        f.id = url;
        l.files.push_back(std::move(f));
    }
    return l;
}

/// writes `head` at the start of a new file of `size` bytes whose rest is a hole: a sparse file, its blocks the head's
void write_sparse(const fs::path& path, const std::string& head, uint64_t size) {
    size = std::max<uint64_t>(size, head.size());
#ifdef _WIN32
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                           nullptr);
    if (h == INVALID_HANDLE_VALUE) throw RemoteError(u8_str(path) + " cannot be written");
    DWORD got = 0;
    const bool sparse = DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &got, nullptr) != 0;
    bool ok = sparse;
    for (size_t at = 0; ok && at < head.size();) {
        const DWORD n = (DWORD) std::min<size_t>(head.size() - at, 1u << 30);
        DWORD wrote = 0;
        ok = WriteFile(h, head.data() + at, n, &wrote, nullptr) && wrote == n;
        at += n;
    }
    LARGE_INTEGER end{};
    end.QuadPart = (LONGLONG) size;
    ok = ok && SetFilePointerEx(h, end, nullptr, FILE_BEGIN) && SetEndOfFile(h);
    CloseHandle(h);
    if (!sparse)
        throw RemoteError(u8_str(path.parent_path()) + ": its drive keeps no sparse files (an NTFS drive does): TEMP "
                          "names the folder lamina writes the files' headers in");
    if (!ok) throw RemoteError(u8_str(path) + " cannot be written");
#else
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) throw RemoteError(u8_str(path) + " cannot be written");
    bool ok = true;
    for (size_t at = 0; ok && at < head.size();) {
        const ssize_t n = ::write(fd, head.data() + at, head.size() - at);
        ok = n > 0;
        if (ok) at += (size_t) n;
    }
    ok = ok && ::ftruncate(fd, (off_t) size) == 0;
    ::close(fd);
    if (!ok) throw RemoteError(u8_str(path) + " cannot be written");
#endif
}

enum class Kind { Gguf, Safetensors, Json, Other };
Kind kind_of(const std::string& path) {
    const std::string n = lower(name_of(path));
    if (ends_with(n, ".gguf")) return Kind::Gguf;
    if (ends_with(n, ".safetensors")) return Kind::Safetensors;
    if (ends_with(n, ".json")) return Kind::Json;
    return Kind::Other;
}

/// the file's size as a reply gives it: Content-Range's total, else a whole body's Content-Length; 0 where it says none
uint64_t reply_size(const HttpReply& r) {
    const std::string range = r.header("content-range");   // bytes 0-1048575/49720539552
    if (const size_t slash = range.rfind('/'); slash != std::string::npos && range.compare(slash + 1, 1, "*") != 0)
        return std::strtoull(range.c_str() + slash + 1, nullptr, 10);
    if (r.status == 200 && !r.header("content-length").empty()) return std::strtoull(r.header("content-length").c_str(), nullptr, 10);
    return 0;
}

}  // namespace

bool is_url(std::string_view s) {
    const std::string l = lower(s.substr(0, 8));
    return starts_with(l, "http://") || starts_with(l, "https://");
}

bool parse_hf_url(const std::string& url, HfTarget& t, const std::string& endpoint) {
    if (!is_url(url)) return false;
    const UrlParts u = split_url(url);
    const std::string ep = endpoint.empty() ? "https://huggingface.co" : std::string(rstrip_chars(endpoint, "/"));
    const std::string host = bare_host(u.host), ep_host = bare_host(split_url(ep).host);
    if (host != "huggingface.co" && host != "hf.co" && host != ep_host) return false;
    t = HfTarget{};
    t.endpoint = ep;
    std::vector<std::string> seg;
    for (const std::string& s : split(u.path, "/"))
        if (!s.empty()) seg.push_back(s);
    auto bad = [&](const std::string& why) {
        throw RemoteError(url + ": " + why + ". lamina reads a repository's URL, a folder's in it "
                          "(.../tree/REVISION/FOLDER) or a file's (.../blob/REVISION/FILE, .../resolve/REVISION/FILE)");
    };
    size_t i = 0;
    static const char* const not_repos[] = {"api", "docs", "blog", "papers", "collections", "organizations", "settings",
                                            "models", "join", "login", "logout", "pricing", "tasks", "learn", "posts",
                                            "chat", "new", "enterprise", "search", "notifications", "billing"};
    if (!seg.empty() && seg[0] == "datasets") {
        t.dataset = true;
        i = 1;
    } else if (!seg.empty() && seg[0] == "spaces") {
        bad("a Space, not a model's repository");
    } else if (!seg.empty() && std::find(std::begin(not_repos), std::end(not_repos), seg[0]) != std::end(not_repos)) {
        bad("not a repository's page");
    }
    if (seg.size() < i + 2) bad("it names no repository (owner/name)");
    t.repo = unquote_to_bytes(seg[i]) + "/" + unquote_to_bytes(seg[i + 1]);
    i += 2;
    t.revision = "main";
    if (i == seg.size()) return true;
    const std::string kind = seg[i++];
    if (kind != "tree" && kind != "blob" && kind != "resolve" && kind != "raw")
        bad("a page of " + t.repo + " that names no folder or file");
    t.file = kind != "tree";
    if (i < seg.size()) {
        t.revision = unquote_to_bytes(seg[i++]);
        // a pull request's or a conversion's ref, its slashes unencoded: refs/pr/N, refs/convert/NAME
        if (t.revision == "refs" && i + 1 < seg.size()) {
            t.revision += "/" + unquote_to_bytes(seg[i]) + "/" + unquote_to_bytes(seg[i + 1]);
            i += 2;
        }
    }
    for (; i < seg.size(); ++i) t.path += (t.path.empty() ? "" : "/") + unquote_to_bytes(seg[i]);
    if (t.file && t.path.empty()) bad("it names no file");
    return true;
}

std::string hf_token() {
    auto env = [](const char* k) {
        const char* v = std::getenv(k);
        return v != nullptr ? std::string(strip(v)) : std::string();
    };
    if (std::string t = env("HF_TOKEN"); !t.empty()) return t;
    auto home_of = [&](std::string p) {   // os.path.expanduser
        if (p == "~" || starts_with(p, "~/") || starts_with(p, "~\\")) {
#ifdef _WIN32
            const std::string home = env("USERPROFILE");
#else
            const std::string home = env("HOME");
#endif
            if (home.empty()) return std::string();
            p = home + p.substr(1);
        }
        return p;
    };
    std::string file = home_of(env("HF_TOKEN_PATH"));
    if (file.empty()) {
        std::string home = home_of(env("HF_HOME"));
        if (home.empty()) {
            const std::string cache = home_of(env("XDG_CACHE_HOME").empty() ? "~/.cache" : env("XDG_CACHE_HOME"));
            if (cache.empty()) return "";
            home = u8_str(u8_path(cache) / "huggingface");
        }
        file = u8_str(u8_path(home) / "token");
    }
    std::ifstream in(u8_path(file), std::ios::binary);
    if (!in) return "";
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return std::string(strip(text));
}

const RemoteFile* RemoteListing::find(const std::string& path) const {
    const auto it = std::lower_bound(files.begin(), files.end(), path,
                                     [](const RemoteFile& f, const std::string& p) { return f.path < p; });
    return it != files.end() && it->path == path ? &*it : nullptr;
}

RemoteListing list_remote(const std::string& url, const std::string& token, const std::string& endpoint) {
    HfTarget t;
    if (parse_hf_url(url, t, endpoint)) return list_hub(t, url, token);
    return list_direct(url);
}

RemoteCopies::RemoteCopies(const RemoteListing& listing, std::string token) : listing_(listing), token_(std::move(token)) {
    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path(ec);
    if (ec) throw RemoteError("no temporary folder (TEMP) for the files' headers: " + ec.message());
    const fs::path root = tmp / ("lamina-check-" + random_hex(8));
    if (!fs::create_directories(root, ec)) throw RemoteError(u8_str(root) + " cannot be made: " + ec.message());
    root_ = u8_str(root);
}

RemoteCopies::~RemoteCopies() {
    std::error_code ec;
    fs::remove_all(u8_path(root_), ec);
}

RemoteCopies::Read RemoteCopies::read_one(const RemoteFile& f) {
    Read out;
    const Kind kind = kind_of(f.path);
    std::string buf;
    bool done = false, foreign = false;
    std::string bad;
    uint64_t next_check = 64 * 1024;
    auto on_data = [&](std::string_view chunk) {
        buf.append(chunk.data(), chunk.size());
        if (kind == Kind::Gguf) {
            if (buf.size() >= 4 && buf.compare(0, 4, "GGUF") != 0) {   // no GGUF (an error's page among them)
                foreign = true;
                return false;
            }
            if (buf.size() >= next_check) {
                try {
                    if (gguf_header_length((const uint8_t*) buf.data(), buf.size()) > 0) {
                        done = true;
                        return false;
                    }
                } catch (const std::exception& e) {
                    bad = e.what();
                    return false;
                }
                if (buf.size() > kHeaderMax) {
                    bad = "a header longer than " + std::to_string(kHeaderMax / kMiB) + " MiB";
                    return false;
                }
                next_check = std::max<uint64_t>(buf.size() + 256 * 1024, buf.size() + buf.size() / 4);
            }
            return true;
        }
        if (kind == Kind::Safetensors && buf.size() >= 8) {
            uint64_t n = 0;
            std::memcpy(&n, buf.data(), 8);
            // (a length the server's reader refuses: the 8 bytes alone tell it, artifact/drafter.hpp)
            if (n < 2 || n > kSafetensorsMax || (buf.size() > 8 && buf[8] != '{') || buf.size() >= 8 + n) {
                done = true;
                return false;
            }
            return true;
        }
        if (kind == Kind::Other) {
            done = true;
            return false;
        }
        return true;   // JSON: whole
    };
    HttpReply r;
    try {
        r = get(f.url, listing_.hub ? listing_.hub_origin : "", token_, {{"Range", "bytes=0-"}},
                kind == Kind::Json ? (size_t) kJsonMax : 0, on_data, [&] {
                    bytes_read_ += buf.size();
                    buf.clear();   // a redirect's body is not the file's
                    done = foreign = false;
                    bad.clear();
                    next_check = 64 * 1024;
                });
    } catch (const HttpError& e) {
        out.error = e.what();
        return out;
    } catch (const RemoteError& e) {
        out.error = e.what();
        return out;
    }
    bytes_read_ += buf.size();
    if (r.status != 200 && r.status != 206) {
        out.error = refusal(r, !token_.empty() && listing_.hub);
        return out;
    }
    if (r.status == 206 && !starts_with(r.header("content-range"), "bytes 0-")) {
        out.error = "the server answered another part of it (" + r.header("content-range") + ")";
        return out;
    }
    out.size = reply_size(r);
    if (f.size != 0 && out.size != 0 && out.size != f.size) {
        out.error = "it is " + std::to_string(out.size) + " bytes, its listing says " + std::to_string(f.size);
        return out;
    }
    if (out.size == 0) out.size = f.size;
    if (out.size == 0 && !done && kind != Kind::Gguf) out.size = buf.size();   // (read whole)
    if (out.size == 0) {
        out.error = "the server does not say its size";
        return out;
    }
    if (kind == Kind::Gguf && !done) {
        if (foreign) {
            out.error = "not a GGUF file (it starts with no GGUF magic)";
            return out;
        }
        if (!bad.empty()) {
            out.error = bad;
            return out;
        }
        try {   // the whole file was read before a check: is its header whole?
            if (gguf_header_length((const uint8_t*) buf.data(), buf.size()) == 0) {
                out.error = buf.size() >= out.size ? "the file ends inside its header" : "the reply ends inside its header";
                return out;
            }
        } catch (const std::exception& e) {
            out.error = e.what();
            return out;
        }
    }
    if (buf.size() > out.size) buf.resize((size_t) out.size);
    out.head = std::move(buf);
    return out;
}

void RemoteCopies::write_copy(const std::string& path, const Read& r) {
    // each part a plain name: none that climbs out of the folder, names a drive or a stream, or a name Windows refuses
    fs::path local = u8_path(root_);
    for (const std::string& part : split(path, "/")) {
        const bool plain = !part.empty() && part != "." && part != ".." && part.back() != '.' && part.back() != ' ' &&
                           std::none_of(part.begin(), part.end(), [](char c) {
                               return (unsigned char) c < 32 || std::strchr("\\:*?\"<>|", c) != nullptr;
                           });
        if (!plain) throw RemoteError(path + ": a file name this PC cannot keep");
        local /= u8_path(part);
    }
    std::error_code ec;
    fs::create_directories(local.parent_path(), ec);
    write_sparse(local, r.head, r.size);
    std::lock_guard<std::mutex> lk(mu_);
    local_[path] = u8_str(local);
}

void RemoteCopies::fetch(const std::vector<std::string>& paths) {
    // the ids to read (once each), and the paths each id's bytes go to
    std::vector<const RemoteFile*> todo;
    std::map<std::string, std::vector<std::string>> paths_of;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const std::string& p : paths) {
            if (local_.count(p) != 0 || error_.count(p) != 0) continue;
            const RemoteFile* f = listing_.find(p);
            if (f == nullptr) {
                error_[p] = "not in the listing";
                continue;
            }
            std::vector<std::string>& to = paths_of[f->id];
            if (std::find(to.begin(), to.end(), p) != to.end()) continue;
            if (to.empty() && by_id_.count(f->id) == 0) todo.push_back(f);
            to.push_back(p);
        }
    }
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next++) < todo.size();) {
            Read r = read_one(*todo[i]);
            if (r.error.empty()) ++files_read_;
            std::lock_guard<std::mutex> lk(mu_);
            by_id_[todo[i]->id] = std::move(r);
        }
    };
    std::vector<std::thread> pool;
    const size_t n = std::min<size_t>(todo.size(), 6);
    for (size_t i = 1; i < n; ++i) pool.emplace_back(work);
    work();
    for (std::thread& t : pool) t.join();
    for (const auto& [id, to] : paths_of) {
        const Read* r = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const auto it = by_id_.find(id);
            if (it != by_id_.end()) r = &it->second;
        }
        for (const std::string& p : to) {
            if (r == nullptr || !r->error.empty()) {
                std::lock_guard<std::mutex> lk(mu_);
                error_[p] = r == nullptr ? "not read" : r->error;
                continue;
            }
            try {
                write_copy(p, *r);
            } catch (const RemoteError& e) {
                std::lock_guard<std::mutex> lk(mu_);
                error_[p] = e.what();
            }
        }
    }
}

bool RemoteCopies::alias(const std::string& as, const std::string& from) {
    const RemoteFile* f = listing_.find(from);
    if (f == nullptr) return false;
    Read r;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = by_id_.find(f->id);
        if (it == by_id_.end() || !it->second.error.empty()) return false;
        r = it->second;
    }
    try {
        write_copy(as, r);
    } catch (const RemoteError&) {
        return false;
    }
    return true;
}

std::string RemoteCopies::local(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = local_.find(path);
    return it == local_.end() ? "" : it->second;
}

std::string RemoteCopies::error(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = error_.find(path);
    return it == error_.end() ? "" : it->second;
}

std::string RemoteCopies::listed(const std::string& local) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& [path, at] : local_)
        if (at == local) return path;
    return "";
}

std::vector<RemoteModel> remote_models(const RemoteListing& l) {
    std::map<std::string, RemoteModel> sets;   // by the first shard's path
    for (const RemoteFile& f : l.files) {
        if (!ends_with(lower(f.path), ".gguf")) continue;
        const std::string name = name_of(f.path), dir = parent_of(f.path);
        int no = 0, count = 0;
        size_t at = 0;
        RemoteModel m;
        if (split_name(name, no, count, at)) {
            for (int i = 1; i <= count; ++i) m.shards.push_back((dir.empty() ? "" : dir + "/") + with_shard(name, at, i));
        } else {
            m.shards.push_back(f.path);
        }
        m.path = m.shards.front();
        sets.emplace(m.path, std::move(m));
    }
    std::vector<RemoteModel> out;
    for (auto& [first, m] : sets) {
        if (!l.focus.empty() && std::find(m.shards.begin(), m.shards.end(), l.focus) == m.shards.end()) continue;
        for (const std::string& s : m.shards) {
            if (const RemoteFile* f = l.find(s)) {
                m.bytes += f->size;
            } else if (m.missing.empty()) {
                m.missing = s;
            }
        }
        out.push_back(std::move(m));
    }
    return out;
}

std::vector<std::string> files_beside(const RemoteListing& l, const std::string& model,
                                      const std::map<std::string, std::string>& config_json) {
    const std::string dir = parent_of(model);
    auto inside = [&](const std::string& p) { return dir.empty() ? p : dir + "/" + p; };
    auto dfly = [&](const std::string& folder) {   // config.cpp's dfly_in: the folder's config.json names DFly's layers
        const auto it = config_json.find(folder.empty() ? "config.json" : folder + "/config.json");
        return it != config_json.end() && it->second.find("\"dfly_target_kv_layers\"") != std::string::npos;
    };
    std::vector<std::string> out;
    for (const RemoteFile& f : l.files) {
        const std::string folder = parent_of(f.path), name = name_of(f.path), n = lower(name);
        if (f.path == model) continue;
        bool want = false;
        if (folder == dir) {
            want = (starts_with(n, "mmproj") && ends_with(n, ".gguf")) ||       // projectors_beside
                   (starts_with(name, "dspark-") && ends_with(name, ".gguf")) ||  // find_dspark
                   name == "mtp.safetensors" || n == lower(kDraftLayerFile) ||   // find_dspark, resolve_draft_layer
                   n == "config.json" ||                                         // find_dfly
                   (ends_with(name, ".safetensors") && dfly(folder));
        } else if (folder == inside("dspark")) {
            want = starts_with(name, "dspark-") && ends_with(name, ".gguf");
        } else if (parent_of(folder) == dir && !folder.empty() && !starts_with(name_of(folder), ".")) {
            // a folder inside the model's (find_dfly)
            want = n == "config.json" || (ends_with(name, ".safetensors") && dfly(folder));
        }
        if (want) out.push_back(f.path);
    }
    return out;
}

std::vector<const RemoteFound*> RemoteModels::models() const {
    std::vector<const RemoteFound*> out;
    for (const RemoteFound& f : found)
        if (f.kind == "model") out.push_back(&f);
    return out;
}

namespace {

/// what a GGUF file is, by its header: a model, or a file a model is run with (a drafter: `found`'s what and read)
std::string gguf_kind(const std::string& local, RemoteFound& found) {
    try {
        const GgufFile g(local);   // (one the reader refuses: a model's row says why)
        (void) g;
    } catch (const std::exception& e) {
        found.error = e.what();
        return "model";
    }
    const GgufRole role = gguf_role(local);
    if (role.kind.empty()) return "model";
    found.what = role.what;
    found.use = role.use;
    found.problem = role.problem;
    found.read = role.problem.empty();
    return role.kind;
}

std::string slurp(const std::string& path) {
    std::ifstream in(u8_path(path), std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

}  // namespace

RemoteModels read_remote_models(const std::string& url) {
    RemoteModels r;
    const auto t0 = std::chrono::steady_clock::now();
    const std::string token = hf_token();
    const char* endpoint = std::getenv("HF_ENDPOINT");
    r.listing = list_remote(url, token, endpoint != nullptr ? endpoint : "");
    r.copies = std::make_unique<RemoteCopies>(r.listing, r.listing.hub ? token : "");
    const std::vector<RemoteModel> sets = remote_models(r.listing);
    if (sets.empty()) {
        size_t safetensors = 0;
        for (const RemoteFile& f : r.listing.files) safetensors += ends_with(lower(f.path), ".safetensors");
        throw RemoteError(url + ": no GGUF file there" +
                          (safetensors > 0 ? " (" + std::to_string(safetensors) + " safetensors files: a checkpoint, "
                                             "which Lamina does not run; its GGUF quants are in other repositories)"
                                           : ""));
    }
    // every first shard, then what each is
    std::vector<std::string> firsts;
    for (const RemoteModel& m : sets)
        if (m.missing != m.path) firsts.push_back(m.path);
    r.copies->fetch(firsts);
    for (const RemoteModel& m : sets) {
        RemoteFound f;
        f.model = m;
        if (m.missing == m.path) {
            f.kind = "model";
        } else if (const std::string e = r.copies->error(m.path); !e.empty()) {
            f.kind = "model";
            f.error = m.path + ": " + e;
        } else {
            f.kind = gguf_kind(r.copies->local(m.path), f);
        }
        r.found.push_back(std::move(f));
    }
    // A split model's shard the listing holds under another name - one table shard for several builds (DeepSeek
    // V4.1's Engram) - by its split keys: its number and count, the model's tensor count, no architecture of its own.
    // Its copy goes under the shard's own name, as a download renames it.
    for (RemoteFound& f : r.found) {
        while (f.kind == "model" && f.error.empty() && !f.model.missing.empty() && f.model.missing != f.model.path) {
            uint64_t total = 0;
            try {
                const GgufFile first(r.copies->local(f.model.path));
                const MetaValue* t = first.get("split.tensors.count");
                total = t != nullptr && t->is_num() ? t->u : 0;
            } catch (const std::exception&) {
                break;
            }
            int no = 0, count = 0;
            size_t at = 0;
            if (total == 0 || !split_name(name_of(f.model.missing), no, count, at)) break;
            const RemoteFound* match = nullptr;
            for (const RemoteFound& o : r.found) {
                const std::string local = o.kind == "model" || o.model.shards.size() != 1 ? "" : r.copies->local(o.model.path);
                if (local.empty()) continue;
                try {
                    const GgufFile g(local);
                    const MetaValue *sno = g.get("split.no"), *scount = g.get("split.count"), *stc = g.get("split.tensors.count");
                    if (sno != nullptr && scount != nullptr && stc != nullptr && g.get("general.architecture") == nullptr &&
                        sno->u + 1 == (uint64_t) no && scount->u == (uint64_t) count && stc->u == total)
                        match = &o;
                } catch (const std::exception&) {
                }
                if (match != nullptr) break;
            }
            if (match == nullptr || !r.copies->alias(f.model.missing, match->model.path)) break;
            const std::string shard = "shard " + std::to_string(no) + " of " + std::to_string(count);
            f.notes.push_back("its " + shard + " is " + match->model.path + " (its split keys): a download keeps it under the "
                              "shard's name");
            if (const RemoteFile* src = r.listing.find(match->model.path)) f.model.bytes += src->size;
            for (RemoteFound& o : r.found)
                if (&o == match) o.notes.push_back(shard + " of " + f.model.path);
            f.model.missing.clear();
            for (const std::string& s : f.model.shards)
                if (r.listing.find(s) == nullptr && r.copies->local(s).empty()) {
                    f.model.missing = s;
                    break;
                }
        }
    }
    // each model's other shards and the files beside it (a folder's config.json first: DFly2's shards are named by it)
    std::vector<std::string> more, configs;
    for (const RemoteFound& f : r.found) {
        if (f.kind != "model" || !f.error.empty()) continue;
        for (size_t i = 1; i < f.model.shards.size(); ++i)
            if (r.listing.find(f.model.shards[i]) != nullptr) more.push_back(f.model.shards[i]);
        for (const std::string& p : files_beside(r.listing, f.model.path, {})) {
            if (ends_with(lower(p), "config.json")) configs.push_back(p);
            else more.push_back(p);
        }
    }
    r.copies->fetch(configs);
    std::map<std::string, std::string> config_json;
    for (const std::string& p : configs)
        if (const std::string local = r.copies->local(p); !local.empty()) config_json[p] = slurp(local);
    for (const RemoteFound& f : r.found)
        if (f.kind == "model" && f.error.empty())
            for (const std::string& p : files_beside(r.listing, f.model.path, config_json))
                if (ends_with(lower(p), ".safetensors")) more.push_back(p);
    r.copies->fetch(more);
    for (RemoteFound& f : r.found) {
        if (f.kind != "model" || !f.error.empty()) continue;
        if (!f.model.missing.empty()) {
            int no = 0, count = 0;
            size_t at = 0;
            split_name(name_of(f.model.missing), no, count, at);
            f.error = "missing model shard " + f.model.missing + " (shard " + std::to_string(no) + " of " +
                      std::to_string(count) + "; is the upload complete?)";
            continue;
        }
        for (const std::string& s : f.model.shards)
            if (const std::string e = r.copies->error(s); !e.empty()) {
                f.error = s + ": " + e;
                break;
            }
    }
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

std::string shown_remote(const RemoteModels& r, std::string_view line) {
    std::string out(line);
    if (r.copies == nullptr) return out;
    // the copies' folder as the download's, in either spelling, and the rest of each path it starts with '/'
    const std::string native = r.copies->root(), generic = replace_all(native, std::string(1, (char) 92), "/");
    static const std::string kDownload = "<download>";
    for (const std::string& root : {native, generic}) {
        for (size_t at = out.find(root); at != std::string::npos; at = out.find(root, at + kDownload.size())) {
            out.replace(at, root.size(), kDownload);
            for (size_t i = at + kDownload.size(); i < out.size() && out[i] != ' ' && out[i] != '"' && out[i] != '\'' &&
                                                    out[i] != ',' && out[i] != ')' && out[i] != '\n';
                 ++i)
                if (out[i] == '\\') out[i] = '/';
        }
    }
    // the model's line: its first shard's URL
    static const std::string kModel = "Model: " + kDownload + "/";
    if (starts_with(out, kModel)) {
        const std::string path = out.substr(kModel.size());
        if (const RemoteFile* f = r.listing.find(path)) out = "Model: " + f->url;
    }
    return out;
}

}  // namespace lamina
