// src/server/remote_model.hpp - lamina-server --check -m URL: the models a URL names, judged from their files' headers.
//
// A Hugging Face URL (a repository, a folder in it, a file) is listed through the Hub's API at one commit; any other
// URL is a GGUF file itself, its split model's other shards beside it by name.  Each file's header is read (one
// request a file, its body read until the header ends: a GGUF's metadata and tensor directory, a safetensors file's
// JSON; a JSON file whole) and written into a temporary folder as a sparse file of the file's size, laid out as a
// download keeps the files.  The server's own checks then read those copies as they read a model on disk: what they
// read of a file is its header (a tensor's data is never read for a check), so a copy answers them as the file would.
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lamina {

/// why a URL names nothing that can be read, said to the user
struct RemoteError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// an http:// or https:// URL
bool is_url(std::string_view s);

/// A Hugging Face URL as the server reads it
struct HfTarget {
    std::string endpoint;   ///< where the Hub's API is asked: https://huggingface.co, or HF_ENDPOINT's
    bool dataset = false;   ///< a dataset's repository (/datasets/...)
    std::string repo;       ///< owner/name
    std::string revision;   ///< as the URL names it; "main" where it names none
    std::string path;       ///< a file or folder in the repository, '/' between folders ("": its root)
    bool file = false;      ///< the URL names a file (blob/, resolve/, raw/), not a folder (the repository, tree/)
};
/// `url` read as a Hugging Face URL: false where its host is none of the Hub's (huggingface.co, hf.co and their www.,
/// HF_ENDPOINT's when `endpoint` names it); RemoteError for one of the Hub's that names no repository, folder or file.
/// `endpoint`: HF_ENDPOINT ("": https://huggingface.co), where the API is asked whichever of the Hub's hosts the URL
/// names.
bool parse_hf_url(const std::string& url, HfTarget& out, const std::string& endpoint = "");

/// The Hub's token as huggingface_hub finds it: HF_TOKEN, else the file HF_TOKEN_PATH names, else the token file of
/// HF_HOME (default: XDG_CACHE_HOME's huggingface folder, ~/.cache/huggingface); "" where there is none
std::string hf_token();

/// One file of a URL's listing
struct RemoteFile {
    std::string path;     ///< from the listing's base ('/' between folders): the repository's path, a direct URL's name
    std::string url;      ///< where its bytes are
    uint64_t size = 0;    ///< as the listing says (a direct URL's: once read)
    std::string sha256;   ///< as the listing says (the Hub's LFS files), else ""
    std::string id;       ///< files of one id hold the same bytes: the sha256, else the Hub's blob id, else the URL
};

/// What a URL names: the files of its scope, at one commit
struct RemoteListing {
    std::string url;      ///< as given
    bool hub = false;     ///< listed through the Hub's API (else: a direct URL)
    std::string hub_origin;   ///< the Hub's scheme://host[:port], to which alone its token goes
    std::string commit;   ///< the Hub's commit the files are read at
    std::string repo;     ///< the Hub's owner/name
    std::string base;     ///< the download URL `RemoteFile::path` is relative to, ending in '/'
    std::string focus;    ///< the path of the file the URL names, "" for a folder
    std::string scope;    ///< the folder of the files listed ("": the base itself)
    std::vector<RemoteFile> files;   ///< the scope's files and its subfolders', by path
    const RemoteFile* find(const std::string& path) const;
};

/// `url`'s listing: through the Hub's API for a Hugging Face URL (its revision as a commit; `token`, where given, sent
/// to the Hub's host alone), else the GGUF file it names and, for a split model's shard, the other shards' URLs by
/// name.  RemoteError where there is no such repository, revision, folder or file, or the Hub refuses them.
RemoteListing list_remote(const std::string& url, const std::string& token, const std::string& endpoint = "");

/// The headers of a listing's files, copied into a temporary folder that goes with it, laid out as a download keeps
/// them: each a sparse file of the file's size holding the bytes read of its start - a GGUF's header, a safetensors
/// file's (an 8-byte length and that much JSON), a JSON file whole (up to 1 MiB) - and nothing more.
class RemoteCopies {
public:
    RemoteCopies(const RemoteListing& listing, std::string token);
    ~RemoteCopies();
    RemoteCopies(const RemoteCopies&) = delete;
    RemoteCopies& operator=(const RemoteCopies&) = delete;

    /// reads those of `paths` (the listing's) not read yet, several at once; one that cannot be read keeps its error
    void fetch(const std::vector<std::string>& paths);
    /// a copy of `from` (read) under the path `as`, which the listing lacks: a split model's shard the listing holds
    /// under another name, as a download would rename it; false where `from` has no copy
    bool alias(const std::string& as, const std::string& from);
    /// the copy of `path` ("" where it has none: not read, or it failed)
    std::string local(const std::string& path) const;
    /// why `path` could not be read ("" where it was, or it was not tried)
    std::string error(const std::string& path) const;
    /// a listing's path from its copy's ("" where `local` is none of the copies)
    std::string listed(const std::string& local) const;
    const std::string& root() const { return root_; }       ///< the copies' folder
    uint64_t bytes_read() const { return bytes_read_; }      ///< over the network so far
    size_t files_read() const { return files_read_; }        ///< the files read so far (one of several paths of one id once)

private:
    struct Read {
        std::string head;   ///< the bytes read of the file's start
        uint64_t size = 0;  ///< the file's
        std::string error;
    };
    Read read_one(const RemoteFile& f);
    void write_copy(const std::string& path, const Read& r);

    const RemoteListing listing_;   // (its own: a RemoteModels that holds it moves)
    std::string token_;
    std::string root_;
    mutable std::mutex mu_;
    std::map<std::string, std::string> local_, error_;   // by the listing's path
    std::map<std::string, Read> by_id_;
    std::atomic<uint64_t> bytes_read_{0};
    std::atomic<size_t> files_read_{0};
};

/// A GGUF model of a listing: a file, or a split model's shards
struct RemoteModel {
    std::string path;                  ///< its first shard's (or its only file's) path in the listing
    std::vector<std::string> shards;   ///< every shard's path, the first first; one the listing lacks named too
    std::string missing;               ///< why it is incomplete: a shard the listing lacks ("": none)
    uint64_t bytes = 0;                ///< its files' size, as the listing says
};
/// The GGUF models of a listing: the one the URL names (its split model's shards), else every one in its scope, by path
std::vector<RemoteModel> remote_models(const RemoteListing& listing);

/// The listing's files the server reads beside a model whose first shard is `model` (a listing path): in its folder the
/// image projectors (mmproj*.gguf), Qwen3.8-Flash-Next's draft layer, Inkling 2's DSpark (dspark-*.gguf, also in a
/// dspark/ folder, and mtp.safetensors), and Hy4-flash's DFly2 (the config.json of its folder or of a folder inside
/// it, the safetensors files beside it) - the files config.cpp and model_config.cpp look for there.  `config_json`:
/// a folder's config.json as read (or "" where it is not read yet): DFly2's shards are named once it is.
std::vector<std::string> files_beside(const RemoteListing& listing, const std::string& model,
                                      const std::map<std::string, std::string>& config_json);

/// What the GGUF files a URL names are, their headers read: every first shard's sorted by its header (`kind`), then of
/// each model its other shards and the files the server reads beside it read too
struct RemoteFound {
    RemoteModel model;
    std::string kind;      ///< "model", or a GgufRole's: "projector", "drafter", "LoRA", "control vector", "engram", ...
    std::string what;      ///< what it is (GgufRole's)
    std::string use;       ///< how Lamina reads it, where it does
    std::string problem;   ///< why Lamina does not read it ("" where it does)
    bool read = false;     ///< Lamina reads it (a projector's and a LoRA's model aside)
    std::vector<std::string> notes;   ///< a split model's shard found under another name, and the like
    std::string error;     ///< why its files could not be read (a model's: a shard missing or refused)
};
struct RemoteModels {
    RemoteListing listing;
    std::unique_ptr<RemoteCopies> copies;
    std::vector<RemoteFound> found;   ///< by path
    double seconds = 0;               ///< the listing and the reads
    /// the models among them (kind "model")
    std::vector<const RemoteFound*> models() const;
};
/// `url`'s GGUF files, read: list_remote with the Hub's token (hf_token, HF_ENDPOINT), RemoteCopies of their headers.
/// RemoteError where the URL names no GGUF file, or nothing of it can be read.
RemoteModels read_remote_models(const std::string& url);

/// a line said with the copies' paths as the download's: the copies' folder as "<download>" with '/' after it, and the
/// model line ("Model: ...") with the first shard's URL
std::string shown_remote(const RemoteModels& r, std::string_view line);

}  // namespace lamina
