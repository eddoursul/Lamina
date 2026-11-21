// src/server/vision.hpp - the resident image encoder (serve/server.py's Vision).
//
// The engine's encoder (include/lamina/vision, on the model's mmproj file) in this process: on a GPU the engine's own
// kernels, on a card the config names, its memory taken before the engine sizes its caches; on the CPU a ggml graph.
// A picture's embeddings are written to a file the engine reads and cached by the picture's hash, so a conversation
// that sends the same picture every turn (most clients do) encodes it once.  A build without the encoder
// (LAMINA_BUILD_VISION=OFF) has the class and refuses to make one.
#pragma once

#include "json.hpp"
#include "sync.hpp"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lamina {

/// an image URL or file is read up to this much (the web app attaches pictures of up to 20 MB)
inline constexpr size_t kImageBytesMax = 32u << 20;
/// the pixels a picture the server decodes may have: Pillow's limit (twice Image.MAX_IMAGE_PIXELS), where it
/// refuses a decompression bomb
inline constexpr uint64_t kImagePixelsMax = 178956970;

class Vision {
public:
    /// the config's "vision" section (mmproj, gpu, cuda_device, threads, max_tokens, min_tokens, q8, lend; its paths
    /// absolute); loaded now unless `lazy`.  std::runtime_error when the build has no encoder, the projector is not
    /// one it reads, or its GPU is not there.
    explicit Vision(const json& cfg, bool lazy = false);
    /// "lend": on a GPU the encoder holds nothing between pictures.  For a request's new pictures it is loaded into
    /// VRAM `lend` gets from the engine (Engine::lend: its expert cache's last slots on that card, which a prompt's
    /// buffers take too) and `unlend` gives back; where the engine lends none, into the card's free VRAM; where
    /// that has no room, the CPU encodes.  Set once the engine exists, and called under the service's turn.
    using Lend = std::function<void*(int device, uint64_t bytes, std::string& why)>;
    void set_lender(Lend lend, std::function<void()> unlend);
    /// whether it borrows ("lend" with a GPU)
    bool lends() const { return lend_; }
    /// a borrowing encoder, the engine running: a picture at the projector's limit through a loan, so that the
    /// first picture's time is a picture's; the log says what the loan took
    void warm_up();
    /// for good (#914): the encoder's memory goes back and its directory (one .sve file per picture) with it
    ~Vision();
    bool alive();
    /// frees the encoder (its RAM, or its VRAM: the process keeps its CUDA context on the card); the encoded
    /// pictures stay cached on disk
    void unload();
    /// loads it again after an unload; the cache is kept
    void restart();
    void close();
    /// a picture's bytes from a data: URL, an http(s) URL or a local file, a URL's or file's at most kImageBytesMax
    /// (std::invalid_argument when it cannot)
    static std::string load(const std::string& source);
    /// an image URL's bytes, at most kImageBytesMax (std::invalid_argument when it cannot be read)
    static std::string download(const std::string& url);
    /// the formats the encoder's decoder reads pass through; any other is converted, a JPEG stored turned (its EXIF
    /// Orientation) is turned upright (std::invalid_argument when it cannot be read, or has over kImagePixelsMax)
    static std::string normalize(const std::string& data);
    /// encode() for every picture of one request: the files of its earlier pictures are never evicted for a later
    /// one (#1072); a request whose client left encodes no more (#1999: ClientGone)
    std::vector<std::pair<std::string, int64_t>> encode_all(const std::vector<std::pair<std::string, bool>>& sources,
                                                            Cancel* cancel);
    /// (its embeddings file, its image tokens); `source`: what load() reads, or the picture's bytes (`is_bytes`);
    /// `keep`: files the caller still needs
    std::pair<std::string, int64_t> encode(const std::string& source, bool is_bytes = false,
                                           const std::vector<std::string>& keep = {});
    std::string dir;   ///< the pictures' files: a path without a space (#480: the GENI line splits at spaces)

private:
    struct Loaded;   ///< an encoder: its GPU kept from the engine's resets, or the loan it is in
    struct Plan;     ///< what the section asks for, settled at the start
    void start();
    /// the encoder for a picture (mu_ held): the resident one, or a borrowing one's lease, made now
    void* encoder_now(std::string& err);
    void lease_end();

    json cfg_;
    std::unique_ptr<Plan> plan_;
    std::unique_ptr<Loaded> loaded_;   ///< the resident encoder; a borrowing one's lease, for a request's pictures
    std::unique_ptr<Loaded> cpu_;      ///< borrowing: the CPU's encoder, once a picture found no VRAM
    bool lend_ = false;
    int holds_ = 0;                    ///< encode_all runs: a lease stays for its pictures
    Lend lender_;
    std::function<void()> unlender_;
    std::atomic<bool> alive_{false};
    std::mutex mu_;   ///< the encoder and the cache: a picture at a time
    std::vector<std::pair<std::string, std::pair<std::string, int64_t>>> cache_;   ///< key -> (file, tokens), oldest first
};

/// whether this build has the image encoder (LAMINA_BUILD_VISION)
bool vision_built();
/// what a build without it says of itself
const char* vision_not_built();
/// a projector file's type (clip.projector_type; "" for a file that names none or cannot be read)
std::string projector_type(const std::string& mmproj);
/// the types the encoder reads: qwen3vl_merger (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3.8-27B), hy_v4_flash and
/// inkling2v (Inkling 2 Mini)
bool projector_type_read(const std::string& type);
/// a path to another computer (\\host\share, //host/share, \\?\UNC\..., \??\UNC\...): Windows would sign in to it
bool network_path(const std::string& path);
/// #874: where one request's combined image-embeddings file goes: on Linux /dev/shm when it has room (the engine
/// reads it back at once), else the encoder's own directory
std::string combined_embeddings_path(const std::string& vision_dir, uintmax_t nbytes);
/// #874: `parts` (files) one after the other into `path`; on Windows as a short-lived file (FILE_ATTRIBUTE_TEMPORARY),
/// which normally never reaches the drive before it is deleted
void write_temporary(const std::string& path, const std::vector<std::string>& parts);
/// a timeout from the environment (LAMINA_ENGINE_STALL_S ...): seconds, nullopt for no limit (0 or less)
std::optional<double> timeout_env(const char* name, double dflt);

}  // namespace lamina
