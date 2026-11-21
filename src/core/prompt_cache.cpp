// src/core/prompt_cache.cpp - the --serve prompt cache (see include/lamina/core/prompt_cache.hpp).
#include "lamina/core/prompt_cache.hpp"

#include "lamina/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace lamina::core {

SequenceCache::SequenceCache(int chain_cap, uint64_t ckpt_bytes, uint64_t park_budget, size_t park_slots,
                             int64_t min_tokens, int sessions)
    : chain_cap_(chain_cap), ckpt_bytes_(ckpt_bytes), lives_((size_t) std::max(1, sessions)),
      parked_(park_budget, park_slots, min_tokens, ckpt_bytes) {}

PromptCache::PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream,
                         int chain_cap, uint64_t park_budget, size_t park_slots, int64_t min_tokens)
    : SequenceCache(chain_cap, session_ckpt_bytes(g), park_budget, park_slots, min_tokens), g_(g), ss_(ss),
      kv_(std::move(kv)), stream_(stream) {}

SequenceCache::~SequenceCache() {
    for (const auto& [p, pinned] : owned_) {
        if (pinned) cudaFreeHost(p);
        else std::free(p);
    }
}

void* SequenceCache::take_buffer() {
    if (!free_.empty()) {
        void* h = free_.back();
        free_.pop_back();
        return h;
    }
    // Pinned for the live chain and its end; past that pageable: slower copies, same result.  Pinned memory is the
    // expert arena's too, and near the driver's limit even a pageable copy can fail.
    void* h = nullptr;
    const bool pinned = pinned_ <= chain_cap_ && cudaMallocHost(&h, (size_t) ckpt_bytes_) == cudaSuccess;
    if (pinned) {
        ++pinned_;
    } else {
        (void) cudaGetLastError();
        h = std::malloc((size_t) ckpt_bytes_);
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
        std::free(o->first);
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
    for (const Part& p : kv_parts(cells)) bytes += p.bytes;
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
    if (cudaMallocHost(&h, (size_t) ckpt_bytes_) != cudaSuccess) {
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

std::vector<PromptCache::Part> PromptCache::kv_parts(int64_t cells) const {
    // [page][kv_head][page_size][head_dim] with an identity page table: the first cells are one prefix of each
    // pool.  The indexer's pooled rows are one per idx_block cells.
    const lamina::kernels::QsaShapes sh = lamina::kernels::qsa_real_shapes();
    const uint64_t row = (uint64_t) g_.idx_key_dim * sizeof(float);
    std::vector<Part> parts;
    for (QsaState* st : kv_) {
        const uint64_t pages = (uint64_t) std::min<int64_t>(st->n_pages, (cells + sh.page_size - 1) / sh.page_size);
        const uint64_t c = pages * (uint64_t) g_.n_head_kv * (uint64_t) sh.page_size;
        for (const KvArray& a : qsa_kv_arrays(*st, g_)) parts.push_back({a.data, c * a.row_bytes});
        const int64_t rows = std::min<int64_t>(cells / sh.idx_block + 2, st->max_cells / sh.idx_block + 2);
        parts.push_back({st->idx_pooled, (uint64_t) rows * row});
    }
    return parts;
}

bool PromptCache::kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) {
    cudaStream_t cs = (cudaStream_t) stream_;
    const std::vector<Part> parts = kv_parts(cells), layout = kv_parts(held);
    for (size_t i = 0; i < parts.size(); ++i) {
        const Part& p = parts[i];
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
    if (!parked_.wants(L)) return;
    if (ensure_kv && !ensure_kv(L)) return;
    const uint64_t bytes = kv_bytes(L);
    // its chain: the checkpoints after `keep_upto` and one at its end, capped
    const bool has_end = live_.find(L) != nullptr;
    int64_t chain = has_end ? 0 : 1;
    for (const Ckpt& c : live_.ckpts) chain += c.pos > keep_upto;
    std::vector<Seq> evicted;
    const uint64_t incoming = bytes + (uint64_t) std::min<int64_t>(chain, chain_cap_) * ckpt_bytes_;
    const int64_t spare = ram_spare ? ram_spare() : ConversationCache::kNoRamLimit;
    if (!parked_.make_room(incoming, evicted, spare)) {
        // logged when the RAM's limit is the lower one: the budget, or a conversation that lends its start, also
        // turn one away (as before the reserve, silently)
        if (incoming <= parked_.budget() && parked_.ram_bound(spare))
            std::fprintf(stderr, "lamina serve: prompt cache: a conversation of %lld cells (%.0f MiB) not parked: the "
                                 "system's available RAM is %.0f MiB %s the reserve\n", (long long) L,
                         (double) incoming / 1048576.0, (double) (spare < 0 ? -spare : spare) / 1048576.0,
                         spare < 0 ? "below" : "above");
        return;
    }
    for (Seq& s : evicted) {
        std::fprintf(stderr, "lamina serve: prompt cache: a parked conversation of %zu cells evicted\n",
                     s.tokens.size());
        release(s, -1);
    }
    evicted.clear();   // their K/V copies go before this one's is allocated
    Seq s;
    s.kv.reset(new (std::nothrow) uint8_t[bytes]);
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
    if (seq.kv && ensure_kv && !ensure_kv(cells)) return false;
    if (seq.kv && !kv_copy(cells, seq.kv_cells, seq.kv.get(), false)) return false;
    return state_load(pos, ck->host);
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
        const bool ok = restore(*donor, best, true);
        donor->pin = -1;
        if (!ok) return zero();
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
    if (!restore(chosen, best, !whole)) {
        release(chosen, -1);
        return zero();
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
    for (Seq& s : evicted) {
        std::fprintf(stderr, "lamina serve: prompt cache: a parked conversation of %zu cells evicted: the system's "
                             "available RAM is %.0f MiB below the reserve\n", s.tokens.size(),
                     (double) -spare / 1048576.0);
        release(s, -1);
    }
    // The evicted chains' buffers went to the free list, which keeps the pinned ones for the next chains: below the
    // reserve no idle buffer is kept, so the RAM the evictions counted goes back to the system.
    for (void* h : free_) {
        const auto o = std::find_if(owned_.begin(), owned_.end(), [&](const auto& b) { return b.first == h; });
        if (o == owned_.end()) continue;
        if (o->second) {
            cudaFreeHost(h);
            --pinned_;
        } else {
            std::free(h);
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
    return s;
}

std::string SequenceCache::summary() const {
    const State s = state();
    char in[64] = "";
    if (lives_.size() > 1) std::snprintf(in, sizeof in, " in %zu of %zu sessions", s.held, lives_.size());
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "live %lld cells%s, %zu checkpoints (%.0f MiB); %zu parked conversations (%.0f MiB, %zu evicted); "
                  "%zu idle checkpoint buffers",
                  (long long) s.live_cells, in, s.checkpoints, (double) s.checkpoints * (double) ckpt_bytes_ / 1048576.0,
                  parked_.size(), (double) parked_.bytes() / 1048576.0, parked_.evictions(), free_.size());
    return buf;
}

}  // namespace lamina::core
