// include/lamina/core/prompt_cache.hpp - the --serve prompt cache.
//
// A request continues from the longest prefix of its prompt that the engine still holds, instead of processing the
// whole conversation again.  Two things hold prefixes:
//
//   * the LIVE conversation: the tokens whose cells the session state holds right now, and its CHECKPOINTS: the
//     fixed-size state at one position.  The K/V cells are indexed by position, so restoring a checkpoint at `pos`
//     and overwriting the cells from `pos` on continues the sequence from there.  An engine that decodes several
//     requests together has a session for each, and every session holds a live conversation: a request runs in the
//     session that holds its conversation, a new conversation in the one whose conversation is the smallest (one
//     that holds nothing first), and one that shares its start with a conversation another session holds copies
//     that start from there;
//   * PARKED conversations: a conversation the live one replaced, kept in host memory with its K/V cells and its own
//     checkpoints, so a client that alternates between conversations (a chat and its title request, an agent and its
//     subagents) does not start over on every switch.  A prompt that continues a parked conversation from its
//     newest checkpoint takes it out of the park; one that matches it only up to an earlier checkpoint (another
//     conversation with the same start: sibling subagents, sessions with one system prompt) borrows the cells up to
//     there, and the parked conversation stays whole - unless the prompt is that conversation's own next request,
//     which a client or a template renders another way than the engine wrote it (superseded, conversation_cache.hpp):
//     then the conversation comes out of the park cut at that checkpoint, with its chain up to there.
//   * the DISK TIER (open_disk, lamina/core/disk_tier.hpp; an engine that copies a range of its cells: kv_range): a
//     conversation the RAM cannot hold is written from the device to a file through a small pinned stage, never
//     whole in RAM, and one that leaves the RAM goes from its copy; read back the same way where a request takes or
//     borrows it.  Matched as any parked one (its tokens and checkpoint positions stay in RAM); its I/O runs in
//     begin() and between requests only.  A
//     conversation taken back keeps its file while it is live: the cells before where the sequence is rewritten
//     stay the file's, its checkpoints there are read when a request needs one, and its next park writes only the
//     cells and checkpoints the file lacks.
//
// Each conversation owns its checkpoints, at most `chain_cap` of them, which thin out geometrically back from its
// newest (chain_victim); a parked conversation leaves only whole (ConversationCache, conversation_cache.hpp).  One
// allocator serves every chain, a free list between uses: the first chain_cap + 1 buffers pinned where the driver
// allows (the live chain and its end), the rest pageable.
//
// SequenceCache is that policy over an engine's own state, which it reaches through five hooks (the fixed-size state
// to and from a buffer, an empty sequence, the K/V cells to and from a parked copy), where the engine has several
// sessions two more (the session the others act on, K/V cells from one session to another), and for the disk tier
// its copies of a range of cells (kv_range and the hooks beside it); PromptCache is
// Flash-Next's: its session state (session_ckpt_*) and the cells of its QSA layers.  The qwen35 path has its own
// (src/qwen35/q36_engine.cu), over its batch slots' sessions, and so has the Inkling 2 path
// (src/inkling2/inkling2_engine.cu).
#pragma once

#include "lamina/core/conversation_cache.hpp"
#include "lamina/core/disk_tier.hpp"
#include "lamina/core/session.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lamina::core {

class SequenceCache {
public:
    /// `chain_cap`: checkpoints a conversation keeps (0: the cache is off); `ckpt_bytes`: one checkpoint's buffer;
    /// `park_budget`, `park_slots`, `min_tokens`: the park's (ConversationCache); `sessions`: the engine's sessions,
    /// each holding one live conversation (one, or as many as requests decode together).
    SequenceCache(int chain_cap, uint64_t ckpt_bytes, uint64_t park_budget, size_t park_slots, int64_t min_tokens,
                  int sessions = 1);
    virtual ~SequenceCache();
    SequenceCache(const SequenceCache&) = delete;
    SequenceCache& operator=(const SequenceCache&) = delete;

    bool enabled() const { return chain_cap_ > 0; }
    /// The disk tier (--conversation-spill-dir): the parked conversations the RAM does not hold, and those that leave
    /// it, in files of a folder of this process's own under `root`, within `budget` bytes (0: min(32 GiB, a quarter
    /// of the disk's free space)) and never into the disk's last DiskTier::kFloor free.  false, with `msg` saying
    /// why: an engine without its range copies (disk_ready), the folder, too little room.  Else `msg` says where.
    bool open_disk(const std::string& root, uint64_t budget, std::string& msg);
    /// --conversation-spill-dir and --conversation-spill-mib (-1: the default budget, 0: none): open_disk where the
    /// cache is on and a folder is named, and its line in the log.
    void open_disk(const std::string& dir, int64_t mib);
    bool disk_on() const { return disk_ != nullptr; }
    /// The elastic K/V (--kv-grow): called before any copy of `cells` K/V cells to or from the device (a parked
    /// conversation's restore, the live one's park), so the pools can grow to hold them first.  false: no room.
    std::function<bool(int64_t cells)> ensure_kv;
    /// --ram-reserve-mib: the RAM the parked conversations may still take, the system's available RAM less the
    /// reserve (negative when it is below it; ConversationCache::kNoRamLimit when it cannot be read).  Unset: only
    /// the park's budget limits them.  The pinned expert arena is not swappable, so with the park's budget beside it
    /// over the RAM the rest of the system swaps instead, and while swap is left the OOM killer does not step in.
    std::function<int64_t()> ram_spare;
    /// An engine whose checkpoint of a prompt's end (checkpoint()'s `end`) keeps its output there: begin() then
    /// continues the same prompt from it with nothing to read.  Unset, one token at least is always processed.
    bool whole_prompt = false;
    /// An engine whose prompts end where a turn ends (no generation prompt after the last message): a prompt end's
    /// checkpoint stays in the chain as any other when the next prompt's end comes, so a request that parts from the
    /// conversation after it (an edited message, an answer in another rendering) continues from there.  Unset, a
    /// conversation keeps its last prompt's end only.
    bool keep_ends = false;
    /// Cells the live sequence holds (a later park reads all of them): the session's the last begin() chose.
    int64_t live_cells() const { return (int64_t) live().tokens.size(); }
    uint64_t ckpt_bytes() const { return ckpt_bytes_; }

    /// Sets the engine's state up for a prompt of `ids`: continues from the longest prefix (at most ids.size() - 1
    /// cells: the last token is always processed) whose token and image prefix matches the live sequence, a
    /// checkpoint or a parked conversation, or zeroes the state.  Image identity is checked before the matching
    /// position is returned; a changed image clamps reuse to before that image.  Text-only requests pass an empty
    /// image-key vector.  `tag`: what else the state depends on (the scales of a request's adapters): a conversation
    /// is continued by requests of its own tag only, and one of another tag starts its own.
    /// With several sessions (choose_start, conversation_cache.hpp): bit s of `busy` says session s decodes, so its
    /// conversation only lends its start; the request runs in `*session`, which the engine has selected on return
    /// (session_select), and the calls below are that session's until the next begin().  -1: every session decodes.
    int64_t begin(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs, uint64_t tag = 0, uint64_t busy = 0,
                  int* session = nullptr);
    /// The cells [pos, pos + n) now hold `tokens`; pos <= live length, and anything after pos is gone.
    void set(int64_t pos, const int64_t* tokens, int64_t n);
    void set(int64_t pos, const int32_t* tokens, int64_t n);
    /// Session `s` took `token` after its conversation's end (a decode step of any session, one that decodes beside
    /// a prompt's read too).
    void append(int s, int64_t token);
    /// Session `s`'s conversation is not continued: its state is no longer what its tokens say.
    void drop(int s);
    /// Session `s` was used now (a request that decoded in it ended): of sessions that hold as much, a new
    /// conversation takes the one used longest ago.
    void touch(int s);
    /// A buffer for the checkpoint a prompt run fills as it goes (prefill::Prefill::set_checkpoint), where the cache
    /// has a pinned one to give (else null: checkpoint() copies at the end).  checkpoint() takes it; one it does not
    /// goes back at the next prepare(), begin() or clear().  With `pageable` (an engine that brings a checkpoint into
    /// any buffer itself): a pinned one where one is idle or can still be pinned, else a pageable one, and which.
    /// `end`: for a prompt's end (checkpoint()'s `end`) - the conversation's last one goes first, and its buffer is
    /// the one to take.
    void* prepare(bool* pageable = nullptr, bool end = false);
    /// Another pinned buffer beside prepare()'s, for a checkpoint the same run fills inside it (prefill::Prefill::
    /// set_marks; null: none to give).  Where none is idle and no more may be pinned, a held checkpoint gives its
    /// own up and moves to a pageable buffer (the conversations' chains hold them all in a session of many).
    /// checkpoint() takes it; one it does not goes back with prepare()'s.
    void* prepare_more();
    /// Between requests: pinned buffers made ready for the next request's run (kSpares idle: its end and one
    /// checkpoint inside it), pinned anew while the cache may pin more (about 0.1 ms a MiB), else given up by a held
    /// checkpoint, which moves to a pageable buffer (prepare_more's way: a parked conversation's, the least recently
    /// used first, else the live one's first); none past the RAM reserve.  That is then no request's time, and the
    /// run's copies go beside it (a pageable buffer waits for the run's end: checkpoint() copies then).
    void spare();
    static constexpr int kSpares = 2;
    /// Checkpoints the state, which is at `pos` == the live length: from `filled`, prepare()'s buffer whose device
    /// state the run has copied in (null: copied now).  `end`: a prompt's end, with the engine's output there
    /// (whole_prompt); a conversation keeps its last one.
    void checkpoint(int64_t pos, void* filled = nullptr, bool end = false);
    /// Forgets the live sequence, every checkpoint and every parked conversation (after non-finite logits: whatever
    /// the state holds may be poisoned).
    void clear();
    /// Between requests: below the RAM reserve (ram_spare), the least recently used parked conversations leave until
    /// the deficit is covered, and the idle checkpoint buffers (pinned ones too) go back to the system.  Returns how
    /// many conversations left.
    size_t shed();

    /// The cache's state: the live conversation, the park's contents, its budgets and its evictions (the CACHE line,
    /// which the Monitor shows).
    struct State {
        int64_t live_cells = 0;                         ///< in every session
        size_t checkpoints = 0;                         ///< of the sessions' conversations
        size_t held = 0;                                ///< sessions that hold a conversation
        size_t parked = 0;                              ///< in RAM and on the disk tier
        uint64_t parked_bytes = 0, parked_budget = 0;   ///< RAM
        size_t slots = 0, evictions = 0;
        size_t disk = 0;   ///< of the parked, those on the disk tier
        uint64_t disk_bytes = 0, disk_budget = 0;
    };
    State state() const;
    /// One line for the log.
    std::string summary() const;

protected:
    // ---- the engine's state
    /// The fixed-size state at `pos` (== the live length) into `host`, a buffer of ckpt_bytes(); `filled`: the run
    /// has copied the device state in already (prepare()'s buffer).  false: it could not be saved.
    virtual bool state_save(int64_t pos, void* host, bool filled) = 0;
    /// The live conversation's tokens (state_save's `pos` of them).
    const std::vector<int64_t>& live_tokens() const { return live().tokens; }
    /// The state `host` holds: the engine is at `pos` then.
    virtual bool state_load(int64_t pos, const void* host) = 0;
    /// An empty sequence.
    virtual void state_zero() = 0;
    /// A parked copy's bytes for `cells` K/V cells.
    virtual uint64_t kv_bytes(int64_t cells) const = 0;
    /// The first `cells` K/V cells between the device and `host`, a parked copy of `held` >= cells cells.
    virtual bool kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) = 0;
    /// A parked copy's buffer `host` is about to be freed: an engine whose kv_copy kept those cells elsewhere (on the
    /// device while its memory is free) forgets them.
    virtual void kv_drop(const uint8_t* host) { (void) host; }
    /// The parked conversation whose copy is `host` leaves the park: the kv_copy to the device that follows, whole or
    /// cut at a checkpoint, is its last (kv_drop comes after).  An engine that left the cells where they were hands
    /// them over to the live sequence instead of copying them.
    virtual void kv_take(const uint8_t* host) { (void) host; }
    /// An engine with several sessions: the hooks above act on session `s` from here.
    virtual void session_select(int s) { (void) s; }
    /// The first `cells` K/V cells of session `from` into the selected one's (a start lent by a conversation that
    /// stays in its session).
    virtual bool kv_from(int from, int64_t cells) {
        (void) from;
        (void) cells;
        return false;
    }
    // ---- the disk tier's way to the K/V (an engine without it has no disk tier: disk_ready)
    virtual bool disk_ready() const { return false; }
    /// Cells [lo, hi) of the live sequence between the device and `host`, laid out as kv_copy lays out a copy of
    /// cells [lo, held) (hi <= held; kv_copy's own is lo = 0).  To the device the ranges come in order, from 0 or
    /// from where a restore starts, the sequence growing as they come.  `lo` a multiple of kv_unit().
    virtual bool kv_range(int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool to_host) {
        (void) lo;
        (void) hi;
        (void) held;
        (void) host;
        (void) to_host;
        return false;
    }
    /// That layout's bytes for cells [lo, held).
    virtual uint64_t kv_range_bytes(int64_t lo, int64_t held) const { return kv_bytes(held - lo); }
    /// Cells [lo, hi) of `copy`, a parked copy of `held` cells, into `out` in kv_range's layout of [lo, hi).
    virtual void kv_pack(int64_t lo, int64_t hi, const uint8_t* copy, int64_t held, uint8_t* out) const {
        (void) lo;
        (void) hi;
        (void) copy;
        (void) held;
        (void) out;
    }
    /// The cells a range starts at a multiple of.
    virtual int64_t kv_unit() const { return 1; }
    /// Every copy into the cache's buffers and every write of the cells has ended (before the disk tier reads them).
    virtual void kv_settle();
    /// The parked copy `host` is to hold all its cells (an engine that left some on the device copies them in).
    virtual bool kv_complete(const uint8_t* host) {
        (void) host;
        return true;
    }

private:
    /// A checkpoint buffer (the header's allocator) and its return to the free list.  An idle buffer past what a
    /// live chain takes goes back to the system at the next begin() (trim) and not when it is given back: an engine
    /// that fills its buffers behind the run (prepare()'s `pageable`) may still owe one its contents until then.
    void* take_buffer();
    void give_back(void* h);
    /// prepare_more()'s pinned buffer where none is idle: a held checkpoint's, whose bytes move to `pageable` (a
    /// parked conversation's, the least recently used first, else one the live conversation held before this
    /// request: its own later ones may still be being written, a split prompt's stages'); null: none is pinned.
    void* pinned_from_held(void* pageable);
    void trim();
    void release(Seq& seq, int64_t keep_upto);   // frees the checkpoints after `keep_upto`
    template <class T> void set_ids(int64_t pos, const T* tokens, int64_t n);   // set()'s, for either id type
    /// Checkpoint i of `seq` out of its chain: its buffer back, its slot in the conversation's file forgotten.
    void drop_ckpt(Seq& seq, size_t i);
    /// Thins `seq`'s chain to the cap (chain_victim).
    void enforce_cap(Seq& seq);
    /// The state at `pos` (== the live length) in a buffer of its own (`filled`: prepare()'s, the device state in it
    /// already); nullptr: none.
    void* save(int64_t pos, void* filled = nullptr);
    /// Parks a copy of the live sequence: its K/V cells, its checkpoints after `keep_upto` (the ones before stay with
    /// the live sequence, which continues from there) and one at its end.
    void stash_live(int64_t keep_upto);
    /// The parked copy `s`'s chain: the live sequence's checkpoints after `keep_upto` moved into it, and one saved at
    /// its end `L` where it has none (`has_end`).  Then, once it is kept, its record: the tokens, the tag and the
    /// pictures that start before L.
    void take_chain(Seq& s, int64_t keep_upto, int64_t L, bool has_end);
    void take_record(Seq& s, int64_t L) const;
    /// The checkpoint at `pos` and, of a parked conversation (`parked`), its K/V cells: all of them, or with `prefix`
    /// only those before `pos` (a borrowed prefix).  A checkpoint in the conversation's file is read into a buffer,
    /// which it keeps (`keep`) or which goes back after the load (a start a parked conversation lends).
    bool restore(Seq& seq, int64_t pos, bool prefix, bool parked, bool keep);

    /// begin()'s choice of the start; begin() keeps it in `begun_`.
    int64_t begin_from(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs, uint64_t tag, uint64_t busy,
                       int* session);
    int64_t begun_ = -1;   // where the request continues the live conversation (-1: before any begin())
    /// Session `s` is the one the request runs in: selected in the engine, and used now.
    void use(int s);
    Seq& live() { return lives_[(size_t) cur_]; }
    const Seq& live() const { return lives_[(size_t) cur_]; }

    // ---- the disk tier (lamina/core/disk_tier.hpp: its folder, its stage, the files)
    /// A conversation the park evicts from RAM (`why`, for the log): to the disk tier where it has room for it,
    /// readmitted there, else gone.
    void evict(Seq&& s, const char* why);
    /// The live sequence, which the RAM does not hold, parked on the disk tier as stash_live parks it in RAM.
    void park_on_disk(int64_t keep_upto);
    /// Room on the disk tier for `bytes` more: within its budget and the disk's free space above its floor beside
    /// the files there, the cells of the files the sessions keep leaving first (but `keep`'s, which a park appends
    /// to), then the least recently used parked files.  fit_lives: the sessions' files within what the parked ones
    /// leave.
    bool disk_room(uint64_t bytes, const Seq* keep = nullptr);
    void fit_lives();
    std::unique_ptr<DiskFile, FileFree> new_file();
    void drop_file(DiskFile* f);
    /// kv_range_bytes as the tier asks for it.
    DiskTier::Bytes range_bytes() const;
    /// Cells [lo, hi) of the live sequence to the end of `f`, in blocks of kv_range's layout.
    bool disk_put_kv(DiskFile& f, int64_t lo, int64_t hi);
    /// A parked copy in RAM (kv_copy's layout of `held` cells) to `f`, in the same blocks.
    bool disk_put_copy(DiskFile& f, const uint8_t* copy, int64_t held);
    /// The first `cells` cells of `f`'s blocks to the device.
    bool disk_get_kv(const DiskFile& f, int64_t cells);
    bool disk_put_ckpt(DiskFile& f, int64_t pos, const void* host);
    bool disk_get_ckpt(const DiskFile& f, int64_t pos, void* host);
    /// The checkpoint of `s` at `pos`, and those up to `upto`, read from its file into buffers of their own where
    /// they hold none.
    bool load_ckpt(Seq& s, int64_t pos);
    bool load_chain(Seq& s, int64_t upto);
    /// A live conversation's file: its cells from `cells` on are rewritten (the file's image ends there); a
    /// checkpoint leaving the chain gives its slot back; the files the sessions keep, their bytes.
    void cut(Seq& s, int64_t cells);
    void forget_slot(Seq& s, int64_t pos);
    /// A session's file gives its cells back (its next park writes them whole), or all of it.
    void drop_cells_of(Seq& s);
    void drop_file_of(Seq& s);
    uint64_t live_files() const;

    int chain_cap_;
    uint64_t ckpt_bytes_;
    uint64_t ckpt_alloc_;   // a checkpoint buffer's bytes: ckpt_bytes_ in whole pages (written to a file whole)
    std::unique_ptr<DiskTier> disk_;
    std::vector<Seq> lives_;   // each session's conversation
    int cur_ = 0;              // the session of the last begin()
    ConversationCache parked_;
    std::vector<void*> free_;
    std::vector<std::pair<void*, bool>> owned_;   // every buffer, and whether it is pinned
    int pinned_ = 0;
    std::vector<void*> prepared_;   // prepare()'s and prepare_more()'s, until checkpoint() takes them
    void drop_prepared();
    bool pinned_buffer(const void* b) const;
    void pinned_first();   // an idle pinned buffer where take_buffer takes the next
    void drop_end();       // the live conversation's checkpoint of a prompt's end, if it has one (keep_ends:
                           // it stays, a checkpoint as the others)
};

/// Qwen3.8-Flash-Next's: the session state and the cells of `kv`.
class PromptCache : public SequenceCache {
public:
    /// `kv`: every state whose cells a parked conversation keeps (the QSA layers and the draft layer).  The rest:
    /// SequenceCache's.
    PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream, int chain_cap,
                uint64_t park_budget, size_t park_slots, int64_t min_tokens);

    /// While a split prompt path runs (prefill/split.hpp): a checkpoint saves the layers below `split_from`
    /// itself and hands its buffer to `split_save`, whose stage writes the rest once that chunk is through it (the
    /// prompt's end waits for that).  0: not split.
    int64_t split_from = 0;
    std::function<void(int64_t pos, void* host)> split_save;

protected:
    bool state_save(int64_t pos, void* host, bool filled) override;
    bool state_load(int64_t pos, const void* host) override;
    void state_zero() override;
    uint64_t kv_bytes(int64_t cells) const override;
    bool kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) override;
    bool disk_ready() const override { return !kv_.empty(); }
    bool kv_range(int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool to_host) override;
    uint64_t kv_range_bytes(int64_t lo, int64_t held) const override;
    void kv_pack(int64_t lo, int64_t hi, const uint8_t* copy, int64_t held, uint8_t* out) const override;
    int64_t kv_unit() const override;

private:
    /// One device region of the K/V cells.
    struct Part {
        void* dev;
        uint64_t bytes;
    };
    /// The device regions of cells [lo, hi) of every K/V state, stored back to back in a parked conversation's
    /// buffer (lo 0) and in the disk tier's blocks: each state's pages of them in every K/V array, then its
    /// indexer's pooled rows (and two past a range's last, as for its end).
    std::vector<Part> kv_parts(int64_t lo, int64_t hi) const;

    const ModelGeometry& g_;
    SessionState& ss_;
    std::vector<QsaState*> kv_;
    void* stream_;
};

}   // namespace lamina::core
