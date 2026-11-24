// src/core/prompt_cache.cpp - the --serve prompt cache (see include/lamina/core/prompt_cache.hpp).
#include "lamina/core/prompt_cache.hpp"

#include "lamina/kernels/qsa.hpp"
#include "lamina/platform/spill.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace lamina::core {

namespace {
using lamina::platform::kSpillAlign;
using lamina::platform::spill_align_up;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
double mib(uint64_t b) { return (double) b / 1048576.0; }

bool disk_failed(const std::string& why) {
    std::fprintf(stderr, "lamina serve: prompt cache: the disk tier: %s\n", why.c_str());
    return false;
}
}  // namespace

// ---- the disk tier
//
// A conversation's file: its K/V cells in blocks, each the K/V states' regions of its cells back to back in
// kv_spans' order (a block of [0, n) is a parked copy in RAM byte for byte), and its checkpoints in slots of a file
// of their own.  Both are read and written unbuffered (lamina/platform/spill.hpp), the cells through the stage: two
// pinned halves in turns, one half's copies to or from the device beside the other's file I/O.
struct SeqFile {
    uint64_t serial = 0;
    std::string kv_path, ck_path;
    struct Block {
        int64_t lo, hi;   ///< its cells [lo, hi)
        uint64_t off;     ///< where it starts in the K/V file
    };
    std::vector<Block> blocks;   ///< lo ascending; a later block's cells over an earlier one's
    uint64_t kv_end = 0;         ///< the K/V file's length, whole pages
    std::vector<std::pair<int64_t, uint32_t>> slots;   ///< a checkpoint's position and its slot
    uint32_t n_slots = 0;
    const uint32_t* slot(int64_t pos) const {
        for (const auto& [p, s] : slots)
            if (p == pos) return &s;
        return nullptr;
    }
};

struct SequenceCache::Disk {
    lamina::platform::SpillDir dir;
    uint64_t budget = 0;   ///< the files' bytes at most
    uint64_t slot = 0;     ///< a checkpoint's slot in its file: a checkpoint buffer, whole pages
    uint64_t half = uint64_t(64) << 20;
    uint8_t* stage[2] = {nullptr, nullptr};
    bool registered[2] = {false, false};      ///< pinned (cudaHostRegister); else pageable, the copies synchronous
    cudaEvent_t done[2] = {nullptr, nullptr};   ///< a half's copies to or from the device have ended
    uint64_t serial = 0;
    ~Disk() {
        for (int h = 0; h < 2; ++h) {
            if (done[h] != nullptr) cudaEventDestroy(done[h]);
            if (registered[h]) cudaHostUnregister(stage[h]);
            lamina::platform::spill_free(stage[h]);
        }
    }
    uint64_t bytes(const SeqFile& f) const { return f.kv_end + (uint64_t) f.n_slots * slot; }
};

bool SequenceCache::open_disk(const std::string& root, uint64_t budget, std::string& msg) {
    if (!enabled()) {
        msg = "the prompt cache is off";
        return false;
    }
    if (kv_spans(0, 1).empty() || kv_stream() == nullptr) {
        msg = "this engine keeps no K/V in files";
        return false;
    }
    auto d = std::make_unique<Disk>();
    int cleaned = 0;
    if (!d->dir.open(root, msg, &cleaned)) return false;
    const uint64_t free = d->dir.free_bytes(), room = free > kSpillFloor ? free - kSpillFloor : 0;
    d->budget = std::min(room, budget > 0 ? budget : std::min<uint64_t>(uint64_t(32) << 30, free / 4));
    d->slot = ckpt_alloc_;
    char b[160];
    if (d->budget < 2 * d->slot) {
        std::snprintf(b, sizeof b, "%.0f MiB free on its disk, %.0f MiB of them past the last %.0f MiB", mib(free),
                      mib(room), mib(kSpillFloor));
        msg = std::string("too little room in ") + root + ": " + b;
        return false;
    }
    for (int h = 0; h < 2; ++h) {
        d->stage[h] = (uint8_t*) lamina::platform::spill_alloc(d->half);
        if (d->stage[h] == nullptr || cudaEventCreateWithFlags(&d->done[h], cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            msg = "its stage could not be allocated";
            return false;
        }
        d->registered[h] = cudaHostRegister(d->stage[h], (size_t) d->half, cudaHostRegisterDefault) == cudaSuccess;
        if (!d->registered[h]) (void) cudaGetLastError();   // pageable: the copies wait for themselves
    }
    std::snprintf(b, sizeof b, ", up to %.0f MiB (%.0f MiB free on its disk)%s", mib(d->budget), mib(free),
                  d->registered[0] && d->registered[1] ? "" : ", through a pageable stage");
    msg = d->dir.path() + b;
    if (cleaned > 0) msg += ", " + std::to_string(cleaned) + " folder" + (cleaned == 1 ? "" : "s") + " of ended engines removed";
    disk_ = std::move(d);
    return true;
}

std::unique_ptr<SeqFile, FileFree> SequenceCache::new_file() {
    std::unique_ptr<SeqFile, FileFree> f(new SeqFile, FileFree{[](void* who, SeqFile* p) {
        static_cast<SequenceCache*>(who)->drop_file(p);
    }, this});
    f->serial = ++disk_->serial;
    f->kv_path = disk_->dir.file(std::to_string(f->serial) + ".kv");
    f->ck_path = disk_->dir.file(std::to_string(f->serial) + ".ckpt");
    return f;
}

void SequenceCache::drop_file(SeqFile* f) {
    lamina::platform::spill_remove(f->kv_path);
    lamina::platform::spill_remove(f->ck_path);
    delete f;
}

bool SequenceCache::disk_room(uint64_t bytes) {
    const uint64_t held = parked_.disk_bytes(), free = disk_->dir.free_bytes();
    // what the disk itself can take beside the files there (its free space unknown: the budget alone)
    const uint64_t room = free == 0 ? UINT64_MAX : held + (free > kSpillFloor ? free - kSpillFloor : 0);
    std::vector<Seq> gone;
    if (!parked_.make_disk_room(bytes, std::min(disk_->budget, room), gone)) return false;
    for (Seq& s : gone) {
        std::fprintf(stderr, "lamina serve: prompt cache: a parked conversation of %zu cells evicted from the disk\n",
                     s.tokens.size());
        release(s, -1);
    }
    return true;
}

bool SequenceCache::disk_put_kv(SeqFile& f, int64_t lo, int64_t hi) {
    Disk& d = *disk_;
    const std::vector<KvSpan> sp = kv_spans(lo, hi);
    uint64_t total = 0;
    for (const KvSpan& s : sp) total += s.bytes;
    const uint64_t padded = spill_align_up(total), windows = (padded + d.half - 1) / d.half;
    cudaStream_t cs = (cudaStream_t) kv_stream();
    std::string err;
    lamina::platform::SpillFile file;
    if (!file.open(f.kv_path, true, err)) return disk_failed(err);
    auto fill = [&](int h, uint64_t w0) {   // the block's bytes [w0, w0 + half) from the device into half h
        const uint64_t w1 = std::min(w0 + d.half, total);
        uint64_t at = 0;
        for (const KvSpan& s : sp) {
            const uint64_t a = std::max(at, w0), b = std::min(at + s.bytes, w1);
            if (a < b && cudaMemcpyAsync(d.stage[h] + (a - w0), (const uint8_t*) s.dev + (a - at), (size_t) (b - a),
                                         cudaMemcpyDeviceToHost, cs) != cudaSuccess)
                return false;
            at += s.bytes;
            if (at >= w1) break;
        }
        return cudaEventRecord(d.done[h], cs) == cudaSuccess;
    };
    bool ok = windows == 0 || fill(0, 0), copied = ok;
    for (uint64_t w = 0; ok && w < windows; ++w) {
        const int h = (int) (w % 2);
        if (w + 1 < windows) ok = copied = fill(1 - h, (w + 1) * d.half);   // (half 1 - h was written last turn)
        ok = ok && (copied = cudaEventSynchronize(d.done[h]) == cudaSuccess) &&
             file.write(f.kv_end + w * d.half, d.stage[h], std::min(d.half, padded - w * d.half), err);
    }
    if (!ok) {
        cudaStreamSynchronize(cs);
        (void) cudaGetLastError();
        return disk_failed(copied ? err : "a copy from the device failed");
    }
    f.blocks.push_back({lo, hi, f.kv_end});
    f.kv_end += padded;
    return true;
}

bool SequenceCache::disk_put_copy(SeqFile& f, const uint8_t* copy, uint64_t bytes, int64_t held) {
    std::string err;
    lamina::platform::SpillFile file;
    const uint64_t padded = spill_align_up(bytes);   // (the copy's buffer is allocated in whole pages)
    if (!file.open(f.kv_path, true, err) || !file.write(f.kv_end, copy, padded, err)) return disk_failed(err);
    f.blocks.push_back({0, held, f.kv_end});
    f.kv_end += padded;
    return true;
}

bool SequenceCache::disk_get_kv(const SeqFile& f, int64_t cells) {
    Disk& d = *disk_;
    cudaStream_t cs = (cudaStream_t) kv_stream();
    std::string err;
    lamina::platform::SpillFile file;
    if (!file.open(f.kv_path, false, err)) return disk_failed(err);
    struct Seg {   // bytes of a block to the device
        uint64_t off;
        uint8_t* dev;
        uint64_t bytes;
    };
    bool used[2] = {false, false}, ok = true, copied = true;
    int turn = 0;
    for (const SeqFile::Block& b : f.blocks) {
        if (b.lo >= cells) break;
        // of each region as the block lays it out (kv_spans(lo, hi)), the first bytes: those of the cells wanted
        const std::vector<KvSpan> lay = kv_spans(b.lo, b.hi), want = kv_spans(b.lo, std::min(b.hi, cells));
        std::vector<Seg> segs;
        uint64_t at = 0, need = 0;
        for (size_t i = 0; i < lay.size(); ++i) {
            if (i < want.size() && want[i].bytes > 0) {
                segs.push_back({at, (uint8_t*) want[i].dev, want[i].bytes});
                need = at + want[i].bytes;
            }
            at += lay[i].bytes;
        }
        const uint64_t end = std::min(spill_align_up(at), spill_align_up(need));
        size_t first = 0;
        for (uint64_t w0 = 0; ok && w0 < end; w0 += d.half) {
            const uint64_t w1 = std::min(w0 + d.half, end);
            while (first < segs.size() && segs[first].off + segs[first].bytes <= w0) ++first;
            if (first == segs.size() || segs[first].off >= w1) continue;   // no bytes wanted here
            const int h = turn;
            turn ^= 1;
            if (used[h]) ok = copied = cudaEventSynchronize(d.done[h]) == cudaSuccess;   // its last copies ended
            ok = ok && file.read(b.off + w0, d.stage[h], w1 - w0, err);
            for (size_t i = first; ok && i < segs.size() && segs[i].off < w1; ++i) {
                const uint64_t a = std::max(segs[i].off, w0), e = std::min(segs[i].off + segs[i].bytes, w1);
                if (a < e)
                    ok = copied = cudaMemcpyAsync(segs[i].dev + (a - segs[i].off), d.stage[h] + (a - w0),
                                                  (size_t) (e - a), cudaMemcpyHostToDevice, cs) == cudaSuccess;
            }
            ok = ok && (copied = cudaEventRecord(d.done[h], cs) == cudaSuccess);
            used[h] = true;
        }
    }
    if (cudaStreamSynchronize(cs) != cudaSuccess) ok = copied = false;
    if (!ok) {
        (void) cudaGetLastError();
        return disk_failed(copied ? err : "a copy to the device failed");
    }
    return true;
}

bool SequenceCache::disk_put_ckpt(SeqFile& f, int64_t pos, const void* host) {
    Disk& d = *disk_;
    std::string err;
    lamina::platform::SpillFile file;
    const uint32_t s = f.n_slots;
    if (!file.open(f.ck_path, true, err)) return disk_failed(err);
    bool ok = true;
    if ((uintptr_t) host % kSpillAlign == 0) {   // the cache's buffers are page-aligned; else through the stage
        ok = file.write((uint64_t) s * d.slot, host, d.slot, err);
    } else {
        for (uint64_t at = 0; ok && at < d.slot; at += d.half) {
            const uint64_t n = std::min(d.half, d.slot - at), m = std::min(n, ckpt_bytes_ - std::min(at, ckpt_bytes_));
            std::memcpy(d.stage[0], (const uint8_t*) host + at, (size_t) m);
            ok = file.write((uint64_t) s * d.slot + at, d.stage[0], n, err);
        }
    }
    if (!ok) return disk_failed(err);
    f.slots.push_back({pos, s});
    ++f.n_slots;
    return true;
}

bool SequenceCache::disk_get_ckpt(const SeqFile& f, int64_t pos, void* host) {
    Disk& d = *disk_;
    const uint32_t* s = f.slot(pos);
    if (s == nullptr) return disk_failed("no checkpoint at " + std::to_string(pos) + " in its file");
    std::string err;
    lamina::platform::SpillFile file;
    if (!file.open(f.ck_path, false, err)) return disk_failed(err);
    bool ok = true;
    if ((uintptr_t) host % kSpillAlign == 0) {
        ok = file.read((uint64_t) *s * d.slot, host, d.slot, err);
    } else {
        for (uint64_t at = 0; ok && at < d.slot; at += d.half) {
            const uint64_t n = std::min(d.half, d.slot - at), m = std::min(n, ckpt_bytes_ - std::min(at, ckpt_bytes_));
            ok = file.read((uint64_t) *s * d.slot + at, d.stage[0], n, err);
            if (ok) std::memcpy((uint8_t*) host + at, d.stage[0], (size_t) m);
        }
    }
    return ok || disk_failed(err);
}

bool SequenceCache::load_chain(Seq& s, int64_t upto) {
    for (Ckpt& c : s.ckpts) {
        if (c.pos > upto || c.host != nullptr) continue;
        void* h = take_buffer();
        if (h == nullptr || !disk_get_ckpt(*s.file, c.pos, h)) {
            give_back(h);
            return false;
        }
        c.host = h;
    }
    return true;
}

void SequenceCache::evict(Seq&& s, const char* why) {
    if (disk_ != nullptr && s.kv) {   // its copy in RAM to the disk tier, then the RAM back
        const auto t0 = Clock::now();
        const uint64_t need = spill_align_up(s.kv_bytes) + s.ckpts.size() * disk_->slot;
        std::unique_ptr<SeqFile, FileFree> f;
        bool ok = disk_room(need) && (f = new_file()) && disk_put_copy(*f, s.kv.get(), s.kv_bytes, s.kv_cells);
        for (size_t i = 0; ok && i < s.ckpts.size(); ++i) ok = disk_put_ckpt(*f, s.ckpts[i].pos, s.ckpts[i].host);
        if (ok) {
            s.kv.reset();
            for (Ckpt& c : s.ckpts) {
                give_back(c.host);
                c.host = nullptr;
            }
            s.disk_bytes = disk_->bytes(*f);
            s.file = std::move(f);
            std::fprintf(stderr, "lamina serve: prompt cache: a parked conversation of %zu cells moved to the disk "
                                 "(%.0f MiB in %.0f ms)%s\n", s.tokens.size(), mib(s.disk_bytes), ms_since(t0), why);
            parked_.readmit(std::move(s));
            return;
        }
    }
    std::fprintf(stderr, "lamina serve: prompt cache: a parked conversation of %zu cells evicted%s\n", s.tokens.size(),
                 why);
    release(s, -1);
}

void SequenceCache::park_on_disk(int64_t keep_upto) {
    Seq& live_ = live();
    const int64_t L = (int64_t) live_.tokens.size();
    const auto t0 = Clock::now();
    // its chain: the checkpoints after `keep_upto` and one at its end, capped (as stash_live's)
    const bool has_end = live_.find(L) != nullptr;
    Seq s;
    for (size_t i = 0; i < live_.ckpts.size();) {
        if (live_.ckpts[i].pos > keep_upto) {
            s.ckpts.push_back(live_.ckpts[i]);
            live_.ckpts.erase(live_.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
    if (!has_end)
        if (void* h = save(L)) s.ckpts.push_back({L, h});   // the session is still at L
    if (s.ckpts.empty()) return;
    enforce_cap(s);
    const uint64_t bytes = kv_bytes(L), need = spill_align_up(bytes) + s.ckpts.size() * disk_->slot;
    std::unique_ptr<SeqFile, FileFree> f;
    const bool room = disk_room(need);
    bool ok = room && (f = new_file()) && cudaDeviceSynchronize() == cudaSuccess && disk_put_kv(*f, 0, L);
    for (size_t i = 0; ok && i < s.ckpts.size(); ++i) ok = disk_put_ckpt(*f, s.ckpts[i].pos, s.ckpts[i].host);
    for (Ckpt& c : s.ckpts) {   // written or not, their buffers go back
        give_back(c.host);
        c.host = nullptr;
    }
    if (!ok) {
        std::fprintf(stderr, "lamina serve: prompt cache: a conversation of %lld cells (%.0f MiB) not parked: %s\n",
                     (long long) L, mib(need), room ? "its file could not be written" : "no room on the disk tier");
        return;
    }
    s.tokens = live_.tokens;
    s.tag = live_.tag;
    for (const ImgKey& image : live_.imgs)
        if (image.start < L) s.imgs.push_back(image);
    s.kv_cells = L;
    s.kv_bytes = bytes;
    s.disk_bytes = disk_->bytes(*f);
    s.file = std::move(f);
    const double ms = ms_since(t0);
    std::fprintf(stderr, "lamina serve: prompt cache: a conversation of %lld cells parked on the disk (%.0f MiB in "
                         "%.0f ms, %.2f GiB/s)\n", (long long) L, mib(s.disk_bytes), ms,
                 ms > 0 ? (double) s.disk_bytes / 1073741824.0 / (ms / 1000.0) : 0.0);
    parked_.park(std::move(s));
}

// ---- the cache

SequenceCache::SequenceCache(int chain_cap, uint64_t ckpt_bytes, uint64_t park_budget, size_t park_slots,
                             int64_t min_tokens, int sessions)
    : chain_cap_(chain_cap), ckpt_bytes_(ckpt_bytes), ckpt_alloc_(spill_align_up(ckpt_bytes)),
      lives_((size_t) std::max(1, sessions)), parked_(park_budget, park_slots, min_tokens, ckpt_bytes) {}

PromptCache::PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream,
                         int chain_cap, uint64_t park_budget, size_t park_slots, int64_t min_tokens)
    : SequenceCache(chain_cap, session_ckpt_bytes(g), park_budget, park_slots, min_tokens), g_(g), ss_(ss),
      kv_(std::move(kv)), stream_(stream) {}

SequenceCache::~SequenceCache() {
    // the conversations' copies, files and chains go while the disk tier is there, before the buffers
    for (Seq& s : parked_.drain()) release(s, -1);
    for (Seq& c : lives_) release(c, -1);
    disk_.reset();
    for (const auto& [p, pinned] : owned_) {
        if (pinned) cudaFreeHost(p);
        else lamina::platform::spill_free(p);
    }
}

void* SequenceCache::take_buffer() {
    if (!free_.empty()) {
        void* h = free_.back();
        free_.pop_back();
        return h;
    }
    // Pinned for the live chain and its end; past that pageable: slower copies, same result.  Pinned memory is the
    // expert arena's too, and near the driver's limit even a pageable copy can fail.  Whole pages either way: the
    // disk tier writes a buffer whole.
    void* h = nullptr;
    const bool pinned = pinned_ <= chain_cap_ && cudaMallocHost(&h, (size_t) ckpt_alloc_) == cudaSuccess;
    if (pinned) {
        ++pinned_;
    } else {
        (void) cudaGetLastError();
        h = lamina::platform::spill_alloc(ckpt_alloc_);
    }
    if (h != nullptr) owned_.emplace_back(h, pinned);
    return h;
}

void SequenceCache::give_back(void* h) {
    if (h != nullptr) free_.push_back(h);
}

void SequenceCache::trim() {
    // idle, a buffer is outside the park's budget: no more are kept than a live chain can take at once (its chain
    // and its end, the pinned ones), the pageable rest goes back to the system
    for (size_t i = 0; free_.size() > (size_t) chain_cap_ + 1 && i < free_.size();) {
        const auto o = std::find_if(owned_.begin(), owned_.end(), [&](const auto& b) { return b.first == free_[i]; });
        if (o == owned_.end() || o->second) {
            ++i;
            continue;
        }
        lamina::platform::spill_free(o->first);
        owned_.erase(o);
        free_.erase(free_.begin() + (ptrdiff_t) i);
    }
}

void SequenceCache::release(Seq& seq, int64_t keep_upto) {
    for (size_t i = 0; i < seq.ckpts.size();) {
        if (seq.ckpts[i].pos > keep_upto) {
            give_back(seq.ckpts[i].host);
            seq.ckpts.erase(seq.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
}

void SequenceCache::enforce_cap(Seq& seq) {
    std::vector<int64_t> pos;
    while ((int64_t) seq.ckpts.size() > chain_cap_) {
        pos.clear();
        for (const Ckpt& c : seq.ckpts) pos.push_back(c.pos);
        const size_t v = chain_victim(pos.data(), pos.size());
        give_back(seq.ckpts[v].host);
        seq.ckpts.erase(seq.ckpts.begin() + (ptrdiff_t) v);
    }
}

void* SequenceCache::save(int64_t pos, void* filled) {
    void* h = filled != nullptr ? filled : take_buffer();
    if (h == nullptr) return nullptr;
    if (!state_save(pos, h, filled != nullptr)) {
        give_back(h);
        return nullptr;
    }
    return h;
}

bool PromptCache::state_save(int64_t pos, void* host, bool filled) {
    const bool split = split_from > 0 && split_save;
    if (filled) {   // the run copied the device state in: the token window is the host's
        session_ckpt_put_window(ss_, g_, host);
    } else {
        cudaDeviceSynchronize();
        if (!(split ? session_ckpt_save_layers(ss_, g_, pos, host, stream_, 0, split_from, 0)
                    : session_ckpt_save(ss_, g_, pos, host, stream_)))
            return false;
    }
    if (split) split_save(pos, host);
    return true;
}

bool PromptCache::state_load(int64_t pos, const void* host) { return session_ckpt_load(ss_, g_, pos, host, stream_); }

void PromptCache::state_zero() {
    session_zero(ss_, g_, nullptr, stream_);
    cudaStreamSynchronize((cudaStream_t) stream_);
}

uint64_t PromptCache::kv_bytes(int64_t cells) const {
    uint64_t bytes = 0;
    for (const KvSpan& p : kv_parts(0, cells)) bytes += p.bytes;
    return bytes;
}

void* SequenceCache::prepare(bool* pageable, bool end) {
    drop_prepared();
    if (!enabled()) return nullptr;
    if (end) drop_end();
    auto pinned = [&](void* b) {
        const auto o = std::find_if(owned_.begin(), owned_.end(), [&](const auto& x) { return x.first == b; });
        return o != owned_.end() && o->second;
    };
    if (pageable != nullptr) {   // an idle pinned buffer goes first (take_buffer takes the list's last)
        const auto f = std::find_if(free_.begin(), free_.end(), pinned);
        if (f != free_.end()) std::iter_swap(f, free_.end() - 1);
    }
    void* h = take_buffer();
    if (h == nullptr) return nullptr;
    const bool pin = pinned(h);
    if (!pin && pageable == nullptr) {   // pageable: a copy into it waits for the run (checkpoint copies then)
        give_back(h);
        return nullptr;
    }
    if (pageable != nullptr) *pageable = !pin;
    prepared_ = h;
    return h;
}

void SequenceCache::spare() {
    if (!enabled() || pinned_ > chain_cap_) return;
    for (void* b : free_) {
        const auto o = std::find_if(owned_.begin(), owned_.end(), [&](const auto& x) { return x.first == b; });
        if (o != owned_.end() && o->second) return;
    }
    void* h = nullptr;
    if (cudaMallocHost(&h, (size_t) ckpt_alloc_) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    ++pinned_;
    owned_.emplace_back(h, true);
    free_.push_back(h);
}

void SequenceCache::drop_end() {
    Seq& live_ = live();
    for (size_t i = 0; i < live_.ckpts.size(); ++i)
        if (live_.ckpts[i].end) {
            if (keep_ends) {
                live_.ckpts[i].end = false;
                continue;
            }
            give_back(live_.ckpts[i].host);
            live_.ckpts.erase(live_.ckpts.begin() + (ptrdiff_t) i);
            return;
        }
}

void SequenceCache::drop_prepared() {
    if (prepared_ == nullptr) return;
    give_back(prepared_);
    prepared_ = nullptr;
}

std::vector<SequenceCache::KvSpan> PromptCache::kv_parts(int64_t lo, int64_t hi) const {
    // [page][kv_head][page_size][head_dim] with an identity page table: a run of pages is one range of each pool.
    // The indexer's pooled rows are one per idx_block cells, and two more past a range's last (as for its end).
    const lamina::kernels::QsaShapes sh = lamina::kernels::qsa_real_shapes();
    const uint64_t row = (uint64_t) g_.idx_key_dim * sizeof(float);
    const uint64_t page = (uint64_t) g_.n_head_kv * (uint64_t) sh.page_size;   // rows a page
    std::vector<KvSpan> parts;
    for (QsaState* st : kv_) {
        const int64_t p0 = std::min<int64_t>(st->n_pages, lo / sh.page_size);
        const int64_t p1 = std::min<int64_t>(st->n_pages, (hi + sh.page_size - 1) / sh.page_size);
        for (const KvArray& a : qsa_kv_arrays(*st, g_))
            parts.push_back({(uint8_t*) a.data + (uint64_t) p0 * page * a.row_bytes,
                             (uint64_t) std::max<int64_t>(0, p1 - p0) * page * a.row_bytes});
        const int64_t rows = st->max_cells / sh.idx_block + 2;
        const int64_t r0 = std::min(rows, lo / sh.idx_block), r1 = std::min(rows, hi / sh.idx_block + 2);
        parts.push_back({(uint8_t*) st->idx_pooled + (uint64_t) r0 * row, (uint64_t) std::max<int64_t>(0, r1 - r0) * row});
    }
    return parts;
}

bool PromptCache::kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) {
    cudaStream_t cs = (cudaStream_t) stream_;
    const std::vector<KvSpan> parts = kv_parts(0, cells), layout = kv_parts(0, held);
    for (size_t i = 0; i < parts.size(); ++i) {
        const KvSpan& p = parts[i];
        const cudaError_t e = to_host ? cudaMemcpyAsync(host, p.dev, (size_t) p.bytes, cudaMemcpyDeviceToHost, cs)
                                      : cudaMemcpyAsync(p.dev, host, (size_t) p.bytes, cudaMemcpyHostToDevice, cs);
        if (e != cudaSuccess) return session_copy_failed(stream_);
        host += layout[i].bytes;
    }
    return cudaStreamSynchronize(cs) == cudaSuccess || session_copy_failed(stream_);
}

void SequenceCache::stash_live(int64_t keep_upto) {
    Seq& live_ = live();
    const int64_t L = (int64_t) live_.tokens.size();
    const bool ram = parked_.wants(L), disk = disk_ != nullptr && L >= parked_.min_tokens();
    if (!ram && !disk) return;
    if (ensure_kv && !ensure_kv(L)) return;
    const uint64_t bytes = kv_bytes(L);
    // its chain: the checkpoints after `keep_upto` and one at its end, capped
    const bool has_end = live_.find(L) != nullptr;
    int64_t chain = has_end ? 0 : 1;
    for (const Ckpt& c : live_.ckpts) chain += c.pos > keep_upto;
    std::vector<Seq> evicted;
    const uint64_t incoming = bytes + (uint64_t) std::min<int64_t>(chain, chain_cap_) * ckpt_bytes_;
    const int64_t spare = ram_spare ? ram_spare() : ConversationCache::kNoRamLimit;
    if (!ram || !parked_.make_room(incoming, evicted, spare)) {
        if (disk) {   // the RAM does not hold it: the disk tier does
            park_on_disk(keep_upto);
            return;
        }
        // logged when the RAM's limit is the lower one: the budget, or a conversation that lends its start, also
        // turn one away (as before the reserve, silently)
        if (incoming <= parked_.budget() && parked_.ram_bound(spare))
            std::fprintf(stderr, "lamina serve: prompt cache: a conversation of %lld cells (%.0f MiB) not parked: the "
                                 "system's available RAM is %.0f MiB %s the reserve\n", (long long) L,
                         (double) incoming / 1048576.0, (double) (spare < 0 ? -spare : spare) / 1048576.0,
                         spare < 0 ? "below" : "above");
        return;
    }
    for (Seq& s : evicted) evict(std::move(s), "");   // to the disk tier where there is one, else gone
    evicted.clear();   // their K/V copies go before this one's is allocated
    Seq s;
    s.kv = std::unique_ptr<uint8_t[], KvFree>((uint8_t*) lamina::platform::spill_alloc(spill_align_up(bytes)),
                                              KvFree{[](void* who, uint8_t* host) {
                                                  static_cast<SequenceCache*>(who)->kv_drop(host);
                                                  lamina::platform::spill_free(host);
                                              }, this});
    if (!s.kv || !kv_copy(L, L, s.kv.get(), true)) return;
    for (size_t i = 0; i < live_.ckpts.size();) {
        if (live_.ckpts[i].pos > keep_upto) {
            s.ckpts.push_back(live_.ckpts[i]);
            live_.ckpts.erase(live_.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
    if (!has_end)
        if (void* h = save(L)) s.ckpts.push_back({L, h});   // the session is still at L
    if (s.ckpts.empty()) return;
    enforce_cap(s);
    s.tokens = live_.tokens;
    s.tag = live_.tag;
    for (const ImgKey& image : live_.imgs)
        if (image.start < L) s.imgs.push_back(image);
    s.kv_cells = L;
    s.kv_bytes = bytes;
    parked_.park(std::move(s));
}

bool SequenceCache::restore(const Seq& seq, int64_t pos, bool prefix) {
    const Ckpt* ck = seq.find(pos);
    if (ck == nullptr) return false;
    const int64_t cells = prefix ? pos : seq.kv_cells;
    const bool copy = seq.kv != nullptr, file = !copy && seq.file != nullptr;
    if ((copy || file) && ensure_kv && !ensure_kv(cells)) return false;
    if (copy && !kv_copy(cells, seq.kv_cells, seq.kv.get(), false)) return false;
    if (file && !disk_get_kv(*seq.file, cells)) return false;
    if (ck->host != nullptr) return state_load(pos, ck->host);
    // a checkpoint in the file (a start lent by a conversation on the disk tier): loaded through a buffer
    void* h = file ? take_buffer() : nullptr;
    const bool ok = h != nullptr && disk_get_ckpt(*seq.file, pos, h) && state_load(pos, h);
    give_back(h);
    return ok;
}

int64_t SequenceCache::begin(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs, uint64_t tag,
                             uint64_t busy, int* session) {
    drop_prepared();
    trim();
    auto zero = [&]() -> int64_t {
        Seq& c = live();
        release(c, 0);
        c.tokens.clear();
        c.imgs.clear();
        c.tag = tag;
        state_zero();
        return 0;
    };
    // (a conversation of another tag holds another state for the same tokens: it is not continued)
    Start st;
    if (enabled()) st = choose_start(lives_, busy, parked_.entries(), ids, imgs, tag, whole_prompt);
    else st.session = cur_;
    if (st.session < 0) return -1;
    if (session != nullptr) *session = st.session;
    if (!enabled()) return zero();
    const int64_t best = st.cells;
    // the session's own conversation gives way to another one: parked whole where the park takes it, then empty
    auto vacate = [&] {
        stash_live(0);
        release(live(), 0);
        live().tokens.clear();
    };
    auto started = [&](const Seq& lender) -> int64_t {   // the session holds the lender's first `best` cells
        Seq& c = live();
        c.tokens.assign(lender.tokens.begin(), lender.tokens.begin() + best);
        c.imgs = imgs;
        c.tag = tag;
        return best;
    };
    switch (st.kind) {
    case Start::kFresh: {
        use(st.session);
        stash_live(0);
        const int64_t result = zero();
        live().imgs = imgs;
        return result;
    }
    case Start::kContinue: {
        // The session's conversation continues: from its end, or from a checkpoint with what came after given up.
        // Another conversation that shares its start and has no other session to take continues in place too, the
        // session's own parked first when the prompt drops more of it than it keeps.
        use(st.session);
        Seq& c = live();
        const int64_t L = (int64_t) c.tokens.size();
        if (st.park_rest) stash_live(best);
        // (the whole prompt: its end's checkpoint brings the engine's output back, also where the state stands there)
        if ((best < L || best == (int64_t) ids.size()) && !restore(c, best)) return zero();
        release(c, best);
        c.tokens.resize((size_t) best);
        c.imgs = imgs;
        return best;
    }
    case Start::kBorrowSession: {
        // Another conversation that shares the start of one a session holds: it takes a session of its own, the
        // cells before `best` copied from the lender's, which stays as it is (it may be decoding).
        Seq& lender = lives_[(size_t) st.from];
        const Ckpt* const ck = lender.find(best);
        touch(st.from);
        use(st.session);
        vacate();
        if (ck == nullptr || !kv_from(st.from, best) || !state_load(best, ck->host)) return zero();
        return started(lender);
    }
    case Start::kBorrowParked: {
        // Another conversation that shares the parked one's start: it borrows the cells before `best`, and the
        // parked conversation stays whole (pinned while the session's own is parked beside it).
        parked_.entries()[(size_t) st.from].pin = best;
        use(st.session);
        vacate();
        auto donor = std::find_if(parked_.entries().begin(), parked_.entries().end(),
                                  [](const Seq& s) { return s.pin >= 0; });
        if (donor == parked_.entries().end()) return zero();
        const auto t0 = Clock::now();
        const bool ok = restore(*donor, best, true);
        donor->pin = -1;
        if (!ok) return zero();
        if (donor->on_disk())
            std::fprintf(stderr, "lamina serve: prompt cache: the first %lld cells of a conversation on the disk "
                                 "lent (%.0f ms)\n", (long long) best, ms_since(t0));
        donor->stamp = parked_.tick();
        return started(*donor);
    }
    case Start::kTake:
        break;
    }
    // The parked conversation continues: the session's own is parked and the chosen one becomes live - from its end,
    // or cut at `best` where its own next request parts from it there (the cells after, and their checkpoints, go).
    const bool whole = best >= parked_.entries()[(size_t) st.from].ckpts.back().pos;
    Seq chosen = parked_.take((size_t) st.from);
    use(st.session);
    vacate();
    if (chosen.kv) kv_take(chosen.kv.get());
    const auto t0 = Clock::now();
    if ((chosen.file && !load_chain(chosen, best)) || !restore(chosen, best, !whole)) {
        release(chosen, -1);
        return zero();
    }
    if (chosen.file) {
        std::fprintf(stderr, "lamina serve: prompt cache: a conversation of %lld cells back from the disk (%.0f ms)\n",
                     (long long) best, ms_since(t0));
        chosen.file.reset();
        chosen.disk_bytes = 0;
    }
    started(chosen);
    live().ckpts = std::move(chosen.ckpts);
    release(live(), best);
    return best;
}

void SequenceCache::set(int64_t pos, const int64_t* tokens, int64_t n) {
    if (!enabled()) return;
    Seq& live_ = live();
    release(live_, pos);
    live_.tokens.resize((size_t) pos);
    live_.tokens.insert(live_.tokens.end(), tokens, tokens + n);
}

void SequenceCache::append(int s, int64_t token) {
    if (enabled()) lives_[(size_t) s].tokens.push_back(token);
}

void SequenceCache::drop(int s) {
    Seq& c = lives_[(size_t) s];
    release(c, -1);
    c.tokens.clear();
    c.imgs.clear();
}

void SequenceCache::touch(int s) { lives_[(size_t) s].stamp = parked_.tick(); }

void SequenceCache::use(int s) {
    cur_ = s;
    session_select(s);
    touch(s);
}

void SequenceCache::set(int64_t pos, const int32_t* tokens, int64_t n) {
    if (!enabled()) return;
    Seq& live_ = live();
    release(live_, pos);
    live_.tokens.resize((size_t) pos);
    for (int64_t i = 0; i < n; ++i) live_.tokens.push_back(tokens[i]);
}

void SequenceCache::checkpoint(int64_t pos, void* filled, bool end) {
    if (filled != nullptr && filled == prepared_) prepared_ = nullptr;   // this call takes it, or gives it back
    else filled = nullptr;
    Seq& live_ = live();
    if (!enabled() || pos <= 0 || pos != (int64_t) live_.tokens.size() || live_.find(pos) != nullptr) {
        if (filled != nullptr) give_back(filled);
        return;
    }
    if (end) drop_end();   // a conversation keeps its last prompt's end: the one before goes
    void* h = save(pos, filled);
    if (h == nullptr) return;
    live_.ckpts.push_back({pos, h, end});   // the newest: positions stay ascending
    enforce_cap(live_);
}

void SequenceCache::clear() {
    drop_prepared();
    for (size_t s = 0; s < lives_.size(); ++s) drop((int) s);
    for (Seq& s : parked_.drain()) release(s, -1);
}

size_t SequenceCache::shed() {
    if (!ram_spare || (parked_.size() == 0 && free_.empty())) return 0;
    const int64_t spare = ram_spare();
    if (spare >= 0 || spare == ConversationCache::kNoRamLimit) return 0;
    std::vector<Seq> evicted;
    const size_t n = parked_.shed(spare, evicted);
    char why[96];
    std::snprintf(why, sizeof why, ": the system's available RAM is %.0f MiB below the reserve", (double) -spare / 1048576.0);
    for (Seq& s : evicted) evict(std::move(s), why);
    // The evicted chains' buffers went to the free list, which keeps the pinned ones for the next chains: below the
    // reserve no idle buffer is kept, so the RAM the evictions counted goes back to the system.
    for (void* h : free_) {
        const auto o = std::find_if(owned_.begin(), owned_.end(), [&](const auto& b) { return b.first == h; });
        if (o == owned_.end()) continue;
        if (o->second) {
            cudaFreeHost(h);
            --pinned_;
        } else {
            lamina::platform::spill_free(h);
        }
        owned_.erase(o);
    }
    free_.clear();
    return n;
}

SequenceCache::State SequenceCache::state() const {
    State s;
    for (const Seq& c : lives_) {
        s.live_cells += (int64_t) c.tokens.size();
        s.checkpoints += c.ckpts.size();
        s.held += !c.tokens.empty();
    }
    s.parked = parked_.size();
    s.parked_bytes = parked_.bytes();
    s.parked_budget = parked_.budget();
    s.slots = parked_.slots();
    s.evictions = parked_.evictions();
    s.disk = parked_.disk_size();
    s.disk_bytes = parked_.disk_bytes();
    s.disk_budget = disk_ ? disk_->budget : 0;
    return s;
}

std::string SequenceCache::summary() const {
    const State s = state();
    char in[64] = "", disk[96] = "";
    if (lives_.size() > 1) std::snprintf(in, sizeof in, " in %zu of %zu sessions", s.held, lives_.size());
    if (disk_) std::snprintf(disk, sizeof disk, ", %zu of them on the disk (%.0f of %.0f MiB)", s.disk, mib(s.disk_bytes),
                             mib(s.disk_budget));
    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "live %lld cells%s, %zu checkpoints (%.0f MiB); %zu parked conversations (%.0f MiB, %zu evicted)%s; "
                  "%zu idle checkpoint buffers",
                  (long long) s.live_cells, in, s.checkpoints, (double) s.checkpoints * (double) ckpt_bytes_ / 1048576.0,
                  s.parked, mib(s.parked_bytes), s.evictions, disk, free_.size());
    return buf;
}

}  // namespace lamina::core
