// include/lamina/core/conversation_cache.hpp - the prompt cache's conversations and their checkpoints, CPU-only.
//
// Every conversation owns its checkpoints: the live one of each session of the engine and each parked one (a
// conversation another request replaced, kept in host memory with its K/V cells) hold chains of their own, so no
// conversation's new checkpoints can take another's.  A chain over its cap gives up the point chain_victim picks; a
// parked conversation leaves only whole, the least recently used first, when the park's bytes (its K/V copies and
// its chains) or its slots run out, or the RAM the system can spare above a reserve, and never while it lends its
// start to a live one.  A request that parts from a conversation before its end is that conversation's own next
// request (superseded), which continues it cut where they part, or another conversation, which borrows its start
// and leaves it whole; choose_start says where a request starts and in which session it runs.
//
// The checkpoint buffers and the K/V copies are opaque here: `SequenceCache` allocates and frees them, this header only
// moves them and counts their bytes, so the policy is testable without a device.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace lamina::core {

/// Identity of an image embedded in a GENI request.  Token IDs alone only describe the image-pad run; the embedding
/// bytes and grid describe what the model actually saw.  `start` is the first image-pad token position.
struct ImgKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& other) const { return start == other.start && hash == other.hash; }
};

/// One checkpoint of a conversation's chain: the session's fixed-size state at `pos`, in a buffer the chain owns.
struct Ckpt {
    int64_t pos = 0;
    void* host = nullptr;   ///< the state's buffer (opaque here: `SequenceCache` owns the allocator)
    /// The state at a prompt's end with the engine's output there: a prompt that ends at `pos` continues from it with
    /// nothing to read (an engine that keeps that output: SequenceCache::whole_prompt).
    bool end = false;
};

/// A parked copy's buffer, and who hears of its end: an engine that keeps the cells elsewhere meanwhile
/// (SequenceCache::kv_drop).
struct KvFree {
    void (*drop)(void* who, const uint8_t* host) = nullptr;
    void* who = nullptr;
    void operator()(uint8_t* p) const {
        if (p != nullptr && drop != nullptr) drop(who, p);
        delete[] p;
    }
};

/// One conversation: its tokens, its checkpoints (positions ascending) and, when parked, a copy of its K/V cells.
struct Seq {
    std::vector<int64_t> tokens;
    std::vector<ImgKey> imgs;
    std::vector<Ckpt> ckpts;
    std::unique_ptr<uint8_t[], KvFree> kv;   ///< cells [0, kv_cells) of every K/V state (parked conversations only)
    int64_t kv_cells = 0;
    uint64_t kv_bytes = 0;
    uint64_t stamp = 0;   ///< the park's clock at its last use (a session's conversation: at its last request)
    /// What its state depends on besides its tokens and images (SequenceCache::begin): requests of another tag do
    /// not continue it.
    uint64_t tag = 0;
    /// begin(): the checkpoint another conversation borrows from this parked one, which no eviction takes while the
    /// live one is parked beside it; -1: none.
    int64_t pin = -1;

    const Ckpt* find(int64_t pos) const {
        for (const Ckpt& c : ckpts)
            if (c.pos == pos) return &c;
        return nullptr;
    }
};

/// The point of a chain over its cap that leaves (`pos` ascending, n >= 2).  Of those between its first (the chain's
/// root, the deepest point later prompts share) and its last, the one whose removal leaves the smallest gap for its
/// distance from the last (upstream's --prompt-cache-spread rule), so the kept points thin out geometrically back from
/// the newest: a prompt that leaves the chain d cells before its end reads about d to 2d of them again.  With two
/// points, the first: a chain of one keeps its newest.
inline size_t chain_victim(const int64_t* pos, size_t n) {
    size_t v = 0;
    double best = 0;
    for (size_t i = 1; i + 1 < n; ++i) {
        const double gap = double(pos[i + 1] - pos[i - 1]) / double(pos[n - 1] - pos[i - 1]);
        if (v == 0 || gap < best) {
            v = i;
            best = gap;
        }
    }
    return v;
}

/// Cells of `seq` a prompt can continue from: the whole sequence when it is a prefix of the prompt (live only: the
/// state is there), else its best checkpoint within the common prefix (its position; -1: none).  Image identity is
/// required besides the tokens: a changed image clamps the match to before that image's first pad token.  One token
/// at least is left to process - but with `ends` a checkpoint of a prompt's end (Ckpt::end) serves the whole prompt.
inline int64_t reusable(const Seq& seq, const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs, bool live,
                        bool ends = false) {
    const int64_t most = (int64_t) ids.size() - 1;
    const int64_t m = std::min<int64_t>((int64_t) seq.tokens.size(), most + (ends ? 1 : 0));
    int64_t lcp = 0;
    while (lcp < m && seq.tokens[(size_t) lcp] == ids[(size_t) lcp]) ++lcp;

    // The token prefix is necessary but not sufficient for multimodal input: every image that falls inside the
    // reusable prefix must have the same embedding/grid identity at the same position.  If an image was changed,
    // removed, added, or reordered, fail closed before its first pad token.
    auto matching_image = [](const std::vector<ImgKey>& all, const ImgKey& wanted) {
        for (const ImgKey& image : all)
            if (image.start == wanted.start) return image == wanted;
        return false;
    };
    for (const ImgKey& image : seq.imgs)
        if (image.start < lcp && !matching_image(imgs, image)) lcp = image.start;
    for (const ImgKey& image : imgs)
        if (image.start < lcp && !matching_image(seq.imgs, image)) lcp = image.start;

    if (live && lcp > 0 && lcp == (int64_t) seq.tokens.size() && lcp <= most) return lcp;
    int64_t best = -1;
    for (const Ckpt& c : seq.ckpts)
        if (c.pos <= lcp && (c.pos <= most || c.end) && c.pos > best) best = c.pos;
    return best;
}

/// A prompt that leaves the live conversation this many cells before its end, and more than it keeps, parks the rest;
/// one that leaves a parked conversation fewer cells before its end takes it back, cut there.
inline constexpr int64_t kParkDropped = 1024;

/// Whether a prompt that parts from the conversation `seq` (a parked one, or one a session holds) at `at`, its end or
/// a checkpoint of it, is that conversation's own next request - so that it continues live from `at`, the cells after
/// given up - and not another conversation with the same start, which borrows the start and leaves it whole.  A
/// conversation that lent its start to its own next request stayed parked beside the one that request grew into, a
/// copy more at every return to it.  It is the conversation's own when
///   * the cells after `at` are few (under kParkDropped): reading them again costs less than keeping that copy;
///   * the prompt is a start of the conversation: the same request again, whose answer is asked for anew;
///   * the prompt holds the end of what the conversation wrote last, in another rendering (a client sends an answer
///     back without its reasoning, a template leaves a block out of the history): a run of kRun of the conversation's
///     last kTail tokens stands in the prompt after `at`, no further on than the conversation is long (and kTail).
inline bool superseded(const Seq& seq, const std::vector<int64_t>& ids, int64_t at) {
    constexpr int64_t kRun = 16, kTail = 64;
    const int64_t L = (int64_t) seq.tokens.size(), n = (int64_t) ids.size();
    if (at < 0 || at > L || at > n) return false;
    if (L - at < kParkDropped) return true;
    if (n <= L && std::equal(ids.begin(), ids.end(), seq.tokens.begin())) return true;
    const auto from = ids.begin() + at, to = ids.begin() + std::min(n, L + kTail);
    for (int64_t s = L - kRun; s >= std::max(at, L - kTail); --s)
        if (std::search(from, to, seq.tokens.begin() + s, seq.tokens.begin() + s + kRun) != to) return true;
    return false;
}

/// Where a request starts (SequenceCache::begin).  An engine holds one conversation in each of its sessions (one
/// session, or as many as requests decode together); a request runs in a session that decodes for no other, and
/// continues from the longest start of its prompt that a session or the park holds.
struct Start {
    enum Kind {
        kFresh,          ///< nothing held is a start of the prompt: an empty sequence
        kContinue,       ///< the conversation `session` holds, from `cells`: its end, or a checkpoint of it
        kBorrowSession,  ///< the first `cells` of the conversation session `from` holds, copied into `session`
        kTake,           ///< the parked conversation `from` comes back into `session`: whole, or cut at `cells`
        kBorrowParked,   ///< the first `cells` of the parked conversation `from`, which stays whole
    };
    Kind kind = kFresh;
    int session = -1;         ///< where the request runs (-1: every session decodes)
    int from = -1;            ///< kBorrowSession: the lending session; kTake, kBorrowParked: the parked entry
    int64_t cells = 0;        ///< the cells the request continues from
    bool park_rest = false;   ///< kContinue: another conversation's request - the session's own is parked first
};

/// The start of a prompt of `ids` among the sessions' conversations `lives` (bit s of `busy`: session s decodes, so
/// it only lends) and the parked ones, of the request's `tag`; `ends` as reusable()'s.  The longest start wins; of
/// equal ones a session that does not decode, then the park, then a decoding session.
///   * A conversation's own next request (superseded: its end, few cells before it, the same request again, its last
///     answer in another rendering) continues it where it is: in its session, or back from the park.
///   * Any other request is a new conversation, which takes the session whose conversation is the smallest (one that
///     holds nothing first; of equal ones the one used longest ago) - a session's conversation that gives way is
///     parked (SequenceCache::begin), which costs by its cells, as bringing it back does, and where the park cannot
///     take it, reading it again does - and starts there from what it shares with another: cells copied from that
///     one's session or from the park.  Where the session it takes is the lender's own, it continues in place, and
///     the lender is parked first when more of it is dropped than kept.
inline Start choose_start(const std::vector<Seq>& lives, uint64_t busy, const std::vector<Seq>& parked,
                          const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs, uint64_t tag, bool ends) {
    const int n = (int) lives.size();
    auto decodes = [&](int s) { return ((busy >> s) & 1) != 0; };
    Start st;
    for (int s = 0; s < n; ++s) {   // a new conversation's session
        if (decodes(s)) continue;
        const Seq& c = lives[(size_t) s];
        const Seq* const now = st.session < 0 ? nullptr : &lives[(size_t) st.session];
        if (now == nullptr || c.tokens.size() < now->tokens.size() ||
            (c.tokens.size() == now->tokens.size() && c.stamp < now->stamp))
            st.session = s;
    }
    if (st.session < 0) return st;
    enum { kNone, kIdle, kParked, kBusy } where = kNone;
    int64_t best = 0;
    int at = -1;
    auto offer = [&](const Seq& c, bool live, int kind, int i) {
        const int64_t r = c.tag == tag ? reusable(c, ids, imgs, live, ends) : -1;
        if (r > best) {
            best = r;
            where = static_cast<decltype(where)>(kind);
            at = i;
        }
    };
    for (int s = 0; s < n; ++s)
        if (!decodes(s)) offer(lives[(size_t) s], true, kIdle, s);
    for (size_t i = 0; i < parked.size(); ++i) offer(parked[i], false, kParked, (int) i);
    for (int s = 0; s < n; ++s)
        if (decodes(s)) offer(lives[(size_t) s], false, kBusy, s);
    if (where == kNone) return st;
    st.cells = best;
    st.from = at;
    if (where == kIdle) {
        const Seq& c = lives[(size_t) at];
        const int64_t dropped = (int64_t) c.tokens.size() - best;
        if (superseded(c, ids, best)) {
            st.kind = Start::kContinue;
            st.session = at;
        } else if (st.session == at) {
            st.kind = Start::kContinue;
            st.park_rest = dropped > best;
        } else {
            st.kind = Start::kBorrowSession;
        }
    } else if (where == kBusy) {
        st.kind = Start::kBorrowSession;
    } else {
        const Seq& c = parked[(size_t) at];
        st.kind = best >= c.ckpts.back().pos || superseded(c, ids, best) ? Start::kTake : Start::kBorrowParked;
    }
    return st;
}

/// The parked conversations: their K/V copies and chains within `budget` bytes, at most `slots` of them (0: no limit),
/// and only conversations of at least `min_tokens` cells.
class ConversationCache {
public:
    ConversationCache(uint64_t budget, size_t slots, int64_t min_tokens, uint64_t ckpt_bytes)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens), ckpt_bytes_(ckpt_bytes) {}

    /// Whether a conversation of `cells` cells is parked at all.
    bool wants(int64_t cells) const { return budget_ > 0 && cells >= min_tokens_; }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    uint64_t budget() const { return budget_; }
    size_t slots() const { return slots_; }
    /// A parked conversation's bytes: its K/V copy and its chain.
    uint64_t bytes(const Seq& s) const { return s.kv_bytes + s.ckpts.size() * ckpt_bytes_; }
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const Seq& s : entries_) n += bytes(s);
        return n;
    }

    /// Makes room for a conversation of `incoming` bytes: the least recently used conversation that lends nothing
    /// leaves whole until it fits and a slot is free.  `spare`: the RAM the park may still take beside what it holds
    /// (the system's available RAM less the reserve; negative when it is below the reserve, kNoRamLimit: only the
    /// budget counts), so with it the park holds at most its bytes now plus `spare`: an eviction gives its bytes back
    /// to the system.  The evicted ones move to `evicted` (their buffers are the caller's to free).  false, with
    /// nothing evicted: it would not fit even so (the budget, the RAM, or only lenders left).
    bool make_room(uint64_t incoming, std::vector<Seq>& evicted, int64_t spare = kNoRamLimit) {
        const uint64_t limit = std::min(budget_, ram_limit(spare));
        if (incoming > limit) return false;
        std::vector<size_t> victims;   // chosen first: an eviction is only committed once the incoming one fits
        uint64_t held = bytes();
        size_t left = entries_.size();
        while ((slots_ > 0 && left >= slots_) || held + incoming > limit) {
            size_t pick = entries_.size();
            for (size_t i = 0; i < entries_.size(); ++i)
                if (entries_[i].pin < 0 && std::find(victims.begin(), victims.end(), i) == victims.end() &&
                    (pick == entries_.size() || entries_[i].stamp < entries_[pick].stamp))
                    pick = i;
            if (pick == entries_.size()) return false;
            victims.push_back(pick);
            held -= bytes(entries_[pick]);
            --left;
        }
        std::sort(victims.begin(), victims.end(), std::greater<size_t>());   // erased from the back
        for (size_t i : victims) {
            evicted.push_back(std::move(entries_[i]));
            entries_.erase(entries_.begin() + (ptrdiff_t) i);
            ++evictions_;
        }
        return true;
    }

    /// Gives RAM back when the system runs below the reserve (`spare` < 0, as make_room's): the least recently used
    /// conversations that lend nothing leave whole until their bytes cover the deficit or none is left.  The evicted
    /// ones move to `evicted`; returns how many.
    size_t shed(int64_t spare, std::vector<Seq>& evicted) {
        size_t n = 0;
        while (spare < 0) {
            size_t pick = entries_.size();
            for (size_t i = 0; i < entries_.size(); ++i)
                if (entries_[i].pin < 0 && (pick == entries_.size() || entries_[i].stamp < entries_[pick].stamp))
                    pick = i;
            if (pick == entries_.size()) break;
            spare += (int64_t) bytes(entries_[pick]);
            evicted.push_back(std::move(entries_[pick]));
            entries_.erase(entries_.begin() + (ptrdiff_t) pick);
            ++evictions_;
            ++n;
        }
        return n;
    }

    /// Admits `s` (after make_room) as the most recently used.
    void park(Seq&& s) {
        entries_.push_back(std::move(s));
        entries_.back().stamp = tick();
    }
    /// Moves the entry at `at` out (the conversation continues live).
    Seq take(size_t at) {
        Seq s = std::move(entries_[at]);
        entries_.erase(entries_.begin() + (ptrdiff_t) at);
        return s;
    }
    /// Moves every entry out (the caller frees their buffers).
    std::vector<Seq> drain() { return std::exchange(entries_, {}); }
    std::vector<Seq>& entries() { return entries_; }
    const std::vector<Seq>& entries() const { return entries_; }
    /// The park's use clock: a parked conversation's stamp advances when it is parked and when it lends its start.
    uint64_t tick() { return ++clock_; }

    /// make_room's `spare` when the RAM is not looked at.
    static constexpr int64_t kNoRamLimit = INT64_MAX;
    /// Whether, with `spare` (as make_room's), the RAM holds the park below its budget.
    bool ram_bound(int64_t spare) const { return ram_limit(spare) < budget_; }

private:
    /// The bytes the park may hold with `spare` RAM to take beside them.
    uint64_t ram_limit(int64_t spare) const {
        if (spare == kNoRamLimit) return UINT64_MAX;
        const int64_t held = (int64_t) bytes();
        if (spare <= -held) return 0;
        return spare > INT64_MAX - held ? UINT64_MAX : (uint64_t) (held + spare);
    }

    uint64_t budget_;
    size_t slots_;
    int64_t min_tokens_;
    uint64_t ckpt_bytes_;
    uint64_t clock_ = 0;
    size_t evictions_ = 0;
    std::vector<Seq> entries_;
};

}  // namespace lamina::core
