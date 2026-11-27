// tests/core/conversation_cache_test.cpp - the prompt cache's conversations, CPU only (conversation_cache.hpp):
//
//   1. ownership: a parked conversation keeps its whole chain while another conversation's chain grows and is
//      thinned beside it, and stays resumable at its end (the pool shared by every conversation let a chatty one
//      take a parked one's checkpoints until it was dropped);
//   2. the park's limits: the slots and the byte budget (K/V copies and chains) evict whole conversations, the least
//      recently used first; no slot limit with 0; a lending conversation is never evicted; nothing is evicted for a
//      conversation that would not fit anyway; short conversations and a budget of 0 park nothing;
//   3. reusable(): the live sequence whole, a parked one from its best checkpoint, a changed image clamping the match;
//      shared_start(), where two prompts part with their pictures compared;
//   4. the RAM: a conversation parks only into the RAM the system can spare above the reserve, evicting the least
//      recently used to make that room, and shed() gives back what the park holds once the system runs below it
//      (a 16 GiB park beside a 101 GiB pinned arena on a 125 GiB PC swapped the system to a halt); ram_bound(),
//      whether the RAM or the budget limits the park;
//   5. superseded(): a parked conversation's own next request (few cells after the checkpoint, the same request
//      again, its last answer sent back in another rendering) against another conversation with the same start;
//   6. choose_start(): where a request starts and in which session it runs - on one session (its conversation
//      continued, parked first only for another conversation that drops a sixteenth of it or more, a parked one
//      taken back or lending) and on several (the session that holds the conversation; a new one in the session
//      that holds nothing, else the one with the smallest conversation, of equal ones the one used longest ago; a
//      sibling's start copied from a session, also one that decodes; never a session that decodes; the same request
//      again beside the first);
//   7. the disk tier: files take neither RAM nor slots, the RAM's evictions and shed() take conversations in RAM
//      only, one readmitted in a file keeps its use and is not counted gone, and make_disk_room() evicts the least
//      recently used files within its limit, never a lender's, nothing for a file that would not fit even so.
#include "lamina/core/conversation_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using lamina::core::chain_victim;
using lamina::core::ConversationCache;
using lamina::core::ImgKey;
using lamina::core::reusable;
using lamina::core::Seq;
using lamina::core::shared_start;
using lamina::core::superseded;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr uint64_t kCkpt = 100;   // a checkpoint's bytes in these tests; a cell's K/V: 10

// a conversation of `cells` tokens (all 7) with checkpoints at `pos`, its K/V copy held
Seq conv(int64_t cells, std::vector<int64_t> pos) {
    Seq s;
    s.tokens.assign((size_t) cells, 7);
    for (int64_t p : pos) s.ckpts.push_back({p, nullptr});
    s.kv_cells = cells;
    s.kv_bytes = (uint64_t) cells * 10;
    return s;
}

std::vector<int64_t> prompt(int64_t cells) { return std::vector<int64_t>((size_t) cells, 7); }

}   // namespace

int main() {
    std::printf("conversation_cache_test\n");
    {   // 1. ownership
        ConversationCache park(1u << 30, 0, 1024, kCkpt);
        std::vector<Seq> evicted;
        Seq a = conv(180000, {16384, 32768, 49152, 179000, 180000});
        check(park.make_room(park.bytes(a), evicted), "a 180K-cell conversation parks");
        park.park(std::move(a));
        std::vector<int64_t> b = {4000};   // another conversation's chain: 200 turns of two checkpoints, 16 kept
        for (int t = 1; t <= 200; ++t)
            for (int64_t p : {4000 + t * 500 - 4, 4000 + t * 500}) {
                b.push_back(p);
                if (b.size() > 16) b.erase(b.begin() + (std::ptrdiff_t) chain_victim(b.data(), b.size()));
            }
        check(b.size() == 16 && b.front() == 4000 && b.back() == 104000,
              "the other chain keeps 16 checkpoints, its root and its newest among them");
        check(park.size() == 1 && park.entries()[0].ckpts.size() == 5 && park.evictions() == 0,
              "the parked conversation keeps all 5 of its checkpoints");
        check(reusable(park.entries()[0], prompt(180500), {}, false) == 180000, "and continues from its end");
    }
    {   // 2. the park's limits
        std::vector<Seq> evicted;
        ConversationCache slots(1u << 30, 2, 0, kCkpt);
        slots.park(conv(1000, {1000}));
        slots.park(conv(2000, {2000}));
        check(slots.make_room(10000, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 1000,
              "two slots: a third conversation evicts the least recently used whole");
        ConversationCache open(1u << 30, 0, 0, kCkpt);
        for (int i = 1; i <= 50; ++i) open.park(conv(1000 * i, {1000 * i}));
        evicted.clear();
        check(open.make_room(10000, evicted) && evicted.empty(), "slots 0: no limit but the budget");
        ConversationCache budget(40000, 0, 0, kCkpt);
        budget.park(conv(1000, {500, 1000}));        // 10,000 + 200
        budget.park(conv(2000, {2000}));             // 20,000 + 100
        budget.entries()[0].stamp = budget.tick();   // the first one used since (it lent its start)
        evicted.clear();
        check(budget.make_room(15000, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 2000,
              "the budget counts the chains: the least recently used leaves, not the oldest parked");
        check(budget.bytes() == 10200 && budget.evictions() == 1, "the park's bytes and evictions follow");
        ConversationCache lend(30000, 0, 0, kCkpt);
        lend.park(conv(1000, {1000}));
        lend.entries()[0].pin = 1000;
        evicted.clear();
        check(!lend.make_room(25000, evicted) && evicted.empty() && lend.size() == 1,
              "a lending conversation is never evicted: the incoming one parks nothing");
        ConversationCache tight(25000, 0, 0, kCkpt);
        tight.park(conv(1000, {1000}));
        tight.entries()[0].pin = 1000;
        tight.park(conv(1200, {1200}));
        evicted.clear();
        check(!tight.make_room(20000, evicted) && evicted.empty() && tight.size() == 2,
              "nothing is evicted when the incoming one would not fit even so");
        check(!tight.make_room(30000, evicted) && tight.size() == 2, "nor for one over the whole budget");
        check(!ConversationCache(1u << 30, 0, 1024, kCkpt).wants(1023) &&
                  ConversationCache(1u << 30, 0, 1024, kCkpt).wants(1024) &&
                  !ConversationCache(0, 0, 0, kCkpt).wants(100000),
              "short conversations, and any with a budget of 0, are not parked");
    }
    {   // 3. reusable()
        const Seq live = conv(5000, {1000, 4000});
        check(reusable(live, prompt(6000), {}, true) == 5000, "the live sequence continues whole");
        check(reusable(live, prompt(6000), {}, false) == 4000, "a parked one from its newest checkpoint");
        std::vector<int64_t> edited = prompt(6000);
        edited[2500] = 8;
        check(reusable(live, edited, {}, true) == 1000, "an edit falls back to the checkpoint before it");
        Seq pics = conv(5000, {1000, 4000});
        pics.imgs = {{2000, 11}};
        check(reusable(pics, prompt(6000), {{2000, 12}}, false) == 1000, "a changed image clamps the match before it");
        check(reusable(pics, prompt(6000), {{2000, 11}}, false) == 4000, "the same image does not");
        // shared_start: where two prompts part (a request's checkpoint where it leaves the one before), their pictures
        // compared besides the ids (of two token types, as the qwen35 path's)
        const std::vector<int32_t> before((size_t) 6000, 7);
        check(shared_start(before, {{2000, 11}}, prompt(6000), {{2000, 11}}, 5999) == 5999,
              "the same picture: the prompts part where their ids do");
        check(shared_start(before, {{2000, 11}}, prompt(6000), {{2000, 12}}, 5999) == 2000,
              "another picture in its place parts them before its first pad");
        check(shared_start(before, {}, prompt(6000), {{2000, 11}}, 5999) == 2000 &&
                  shared_start(before, {{2000, 11}}, prompt(6000), {}, 5999) == 2000,
              "so does a picture added or removed");
        check(shared_start(before, {{2000, 11}, {3000, 13}}, prompt(6000), {{2000, 11}, {3100, 13}}, 5999) == 3000,
              "a picture moved: before the earlier of its places");
        check(shared_start(before, {}, edited, {}, 5999) == 2500 &&
                  shared_start(before, {}, prompt(4000), {}, 5999) == 4000,
              "without pictures: the first id that differs, or the shorter prompt's end");
        // a prompt's end: its checkpoint serves the same prompt whole where the engine keeps its output there
        Seq ended = conv(5200, {4000, 5000, 5200});
        ended.ckpts[1].end = true;
        check(reusable(ended, prompt(5000), {}, false) == 4000 && reusable(ended, prompt(5000), {}, true) == 4000,
              "the same prompt again: a token is left to process without `ends`");
        check(reusable(ended, prompt(5000), {}, false, true) == 5000 &&
                  reusable(ended, prompt(5000), {}, true, true) == 5000,
              "... and none with it: the checkpoint of its end");
        check(reusable(ended, prompt(5200), {}, false, true) == 5000 &&
                  reusable(ended, prompt(5200), {}, true, true) == 5000,
              "a checkpoint that is no prompt's end does not serve a whole prompt");
        check(reusable(ended, prompt(6000), {}, false, true) == 5200 &&
                  reusable(ended, prompt(6000), {}, true, true) == 5200,
              "a longer prompt continues past it as before");
    }
    {   // 4. the RAM
        std::vector<Seq> evicted;
        auto two = [] {   // a budget that never binds; the least recently used first: 10,100 then 20,100 bytes
            ConversationCache c(1u << 30, 0, 0, kCkpt);
            c.park(conv(1000, {1000}));
            c.park(conv(2000, {2000}));
            return c;
        };
        ConversationCache open = two();
        check(open.make_room(50000, evicted) && evicted.empty(), "no RAM limit given: only the budget counts");
        ConversationCache room = two();
        check(room.make_room(5000, evicted, 5000) && evicted.empty(),
              "a conversation the spare RAM holds evicts nothing");
        ConversationCache tight = two();
        check(tight.make_room(15000, evicted, 6000) && evicted.size() == 1 && evicted[0].kv_cells == 1000 &&
                  tight.size() == 1,
              "one the spare RAM does not hold evicts the least recently used until it does");
        ConversationCache none = two();
        evicted.clear();
        check(!none.make_room(40000, evicted, 5000) && evicted.empty() && none.size() == 2,
              "nothing is evicted for one the spare RAM would not hold even with the park empty");
        ConversationCache below = two();
        evicted.clear();
        check(below.make_room(5000, evicted, -6000) && evicted.size() == 2 && below.size() == 0,
              "below the reserve one parks only in place of older ones that cover the deficit as well");
        ConversationCache short_ram = two();
        evicted.clear();
        check(!short_ram.make_room(5000, evicted, -30000) && evicted.empty() && short_ram.size() == 2,
              "and none parks when the whole park could not cover it");
        ConversationCache shed = two();
        check(shed.shed(0, evicted) == 0 && shed.shed(100, evicted) == 0 && evicted.empty() && shed.size() == 2,
              "at or above the reserve shed() keeps every conversation");
        check(shed.shed(-5000, evicted) == 1 && evicted.size() == 1 && evicted[0].kv_cells == 1000 &&
                  shed.size() == 1 && shed.evictions() == 1,
              "below it the least recently used leave until the deficit is covered");
        ConversationCache all = two();
        evicted.clear();
        check(all.shed(-1000000, evicted) == 2 && evicted.size() == 2 && all.size() == 0 && all.bytes() == 0,
              "a deficit larger than the park empties it");
        ConversationCache lend = two();
        lend.entries()[0].pin = 1000;
        evicted.clear();
        check(lend.shed(-1000000, evicted) == 1 && lend.size() == 1 && lend.entries()[0].pin == 1000,
              "a lending conversation is never shed");
        ConversationCache bound(40000, 0, 0, kCkpt);   // 30,200 bytes held
        bound.park(conv(1000, {1000}));
        bound.park(conv(2000, {2000}));
        check(!bound.ram_bound(ConversationCache::kNoRamLimit) && bound.ram_bound(5000) && !bound.ram_bound(9800),
              "the RAM holds the park below its budget only where held + spare is less (a refusal is logged then)");
    }
    {   // 5. a parked conversation's own next request, or another conversation with its start
        // a 20,000-cell conversation: a shared start (tokens 7), a checkpoint at 16,000, then 4,000 cells of its own:
        // a prompt's tail (3) and an answer the engine wrote (reasoning 5, then the answer's 100 .. 1,999)
        Seq own = conv(16000, {4000, 16000, 20000});
        for (int i = 0; i < 100; ++i) own.tokens.push_back(3);
        for (int i = 0; i < 1900; ++i) own.tokens.push_back(5);
        for (int i = 0; i < 2000; ++i) own.tokens.push_back(100 + i);
        own.kv_cells = (int64_t) own.tokens.size();
        // its next request: the same start and tail, the answer without its reasoning, a new turn
        std::vector<int64_t> next(own.tokens.begin(), own.tokens.begin() + 16100);
        for (int i = 0; i < 2000; ++i) next.push_back(100 + i);
        for (int i = 0; i < 300; ++i) next.push_back(9);
        check(superseded(own, next, 16000),
              "the answer sent back without its reasoning: the conversation's own request");
        std::vector<int64_t> cut = next;   // ... its last two tokens not sent back
                                           // (a draft or two past the client's text)
        cut.erase(cut.begin() + 16100 + 1998, cut.begin() + 16100 + 2000);
        check(superseded(own, cut, 16000), "... with the answer's last tokens missing");
        std::vector<int64_t> again(own.tokens.begin(), own.tokens.begin() + 16100);
        check(superseded(own, again, 16000), "the same request again: its own");
        std::vector<int64_t> sibling(own.tokens.begin(), own.tokens.begin() + 16000);
        for (int i = 0; i < 6000; ++i) sibling.push_back(50000 + i);
        check(!superseded(own, sibling, 16000), "another conversation with the same 16,000 cells: not its own");
        check(!superseded(own, sibling, 4000), "... nor from the earlier checkpoint");
        std::vector<int64_t> quote = sibling;   // 12 of the answer's last tokens stand in it: under the run
        for (int i = 0; i < 12; ++i) quote[(size_t) (17000 + i)] = 100 + 1988 + i;
        check(!superseded(own, quote, 16000), "a run shorter than the rule's does not make it its own");
        Seq brief = conv(16000, {4000, 16000, 16900});   // 900 cells after the checkpoint: fewer than kParkDropped
        for (int i = 0; i < 900; ++i) brief.tokens.push_back(100 + i);
        check(superseded(brief, sibling, 16000), "few cells after the checkpoint: taken back whoever asks");
        check(!superseded(brief, sibling, 4000), "... but not from a checkpoint far before its end");
        check(!superseded(own, sibling, 30000) && !superseded(own, sibling, -1), "a place outside either is none");
    }
    {   // 6. where a request starts: one session, then several
        using lamina::core::choose_start;
        using lamina::core::Start;
        // a 20,000-cell conversation with 16,000 cells of start (tokens 7) and 4,000 of its own (its number)
        auto talk = [](int64_t own, uint64_t stamp) {
            Seq s = conv(16000, {4000, 16000, 20000});
            for (int i = 0; i < 4000; ++i) s.tokens.push_back(own + i);
            s.stamp = stamp;
            return s;
        };
        auto next_of = [](const Seq& s, int64_t more) {   // the conversation's next request: all of it and a turn
            std::vector<int64_t> ids = s.tokens;
            for (int i = 0; i < more; ++i) ids.push_back(9);
            return ids;
        };
        std::vector<int64_t> sibling = prompt(16000);   // another conversation with the same 16,000 cells
        for (int i = 0; i < 6000; ++i) sibling.push_back(50000 + i);
        std::vector<int64_t> stranger(5000, 8);
        const std::vector<Seq> none;
        auto seqs = [](Seq a, Seq b = Seq{}, bool two = false) {   // (a Seq holds its K/V copy: moved, not copied)
            std::vector<Seq> v;
            v.push_back(std::move(a));
            if (two) v.push_back(std::move(b));
            return v;
        };

        std::vector<Seq> one = seqs(talk(1000, 1));
        Start st = choose_start(one, 0, none, next_of(one[0], 300), {}, 0, false);
        check(st.kind == Start::kContinue && st.session == 0 && st.cells == 20000 && !st.park_rest,
              "one session: its conversation's next request continues it from its end");
        st = choose_start(one, 0, none, sibling, {}, 0, false);
        check(
            st.kind == Start::kContinue && st.session == 0 && st.cells == 16000 && st.park_rest,
            "one session: a sibling continues in place, the conversation parked first (it drops 4,000 of its 20,000)");
        std::vector<int64_t> early = prompt(4000);   // ... one that shares 4,000 cells only
        for (int i = 0; i < 9000; ++i) early.push_back(60000 + i);
        st = choose_start(one, 0, none, early, {}, 0, false);
        check(st.kind == Start::kContinue && st.cells == 4000 && st.park_rest,
              "one session: one that drops more than it keeps has the conversation parked first too");
        Seq deep = conv(16000, {4000, 16000, 18800, 20000});   // a checkpoint 1,200 cells before its end
        for (int i = 0; i < 4000; ++i) deep.tokens.push_back(1000 + i);
        std::vector<int64_t> rewound(deep.tokens.begin(), deep.tokens.begin() + 18800);
        for (int i = 0; i < 1500; ++i) rewound.push_back(60000 + i);
        std::vector<Seq> long_one = seqs(std::move(deep));
        st = choose_start(long_one, 0, none, rewound, {}, 0, false);
        check(st.kind == Start::kContinue && st.cells == 18800 && !st.park_rest,
              "one session: another request that drops less than a kParkShare-th (1,200 of 20,000) parks nothing");
        Seq written = conv(4000, {4000});   // 6,000 cells written after a start of 4,000: the last 5,000 an answer
        for (int i = 0; i < 6000; ++i) written.tokens.push_back(100 + i);
        std::vector<int64_t> rendered = prompt(4000);   // its next request: the answer without the 1,000 before it
        for (int i = 1000; i < 6000; ++i) rendered.push_back(100 + i);
        for (int i = 0; i < 200; ++i) rendered.push_back(9);
        std::vector<Seq> wrote = seqs(std::move(written));
        st = choose_start(wrote, 0, none, rendered, {}, 0, false);
        check(st.kind == Start::kContinue && st.cells == 4000 && !st.park_rest,
              "one session: its own next request in another rendering parks no copy of what it drops");
        st = choose_start(one, 0, none, stranger, {}, 0, false);
        check(st.kind == Start::kFresh && st.session == 0,
              "one session: a conversation that shares nothing starts empty");
        st = choose_start(one, 1, none, stranger, {}, 0, false);
        check(st.session < 0, "one session that decodes: no place for a request");
        std::vector<Seq> parked = seqs(talk(2000, 1));
        st = choose_start(one, 0, parked, next_of(parked[0], 300), {}, 0, false);
        check(st.kind == Start::kTake && st.from == 0 && st.session == 0 && st.cells == 20000,
              "a parked conversation's next request takes it back");
        std::vector<Seq> blank = seqs(Seq{});
        st = choose_start(blank, 0, parked, sibling, {}, 0, false);
        check(st.kind == Start::kBorrowParked && st.from == 0 && st.cells == 16000,
              "a sibling of a parked conversation borrows its start");

        // two sessions: conversation A (used first) and conversation B
        std::vector<Seq> two = seqs(talk(1000, 1), talk(2000, 2), true);
        st = choose_start(two, 0, none, next_of(two[1], 300), {}, 0, false);
        check(st.kind == Start::kContinue && st.session == 1 && st.cells == 20000,
              "several sessions: a request runs in the session that holds its conversation");
        st = choose_start(two, 0, none, stranger, {}, 0, false);
        check(st.kind == Start::kFresh && st.session == 0,
              "a new conversation takes the session used longest ago where they hold as much");
        std::vector<Seq> sizes = seqs(talk(1000, 1), conv(6000, {6000}), true);   // 20,000 cells used first, 6,000 last
        sizes[1].stamp = 2;
        st = choose_start(sizes, 0, none, stranger, {}, 0, false);
        check(st.kind == Start::kFresh && st.session == 1,
              "... and the one whose conversation is the smaller, whichever was used last");
        std::vector<Seq> gap = seqs(talk(1000, 1), Seq{}, true);
        st = choose_start(gap, 0, none, stranger, {}, 0, false);
        check(st.kind == Start::kFresh && st.session == 1, "... or the one that holds nothing");
        st = choose_start(gap, 0, none, sibling, {}, 0, false);
        check(st.kind == Start::kBorrowSession && st.from == 0 && st.session == 1 && st.cells == 16000,
              "a sibling takes a session of its own with the start copied: the lender stays");
        st = choose_start(two, 0, none, sibling, {}, 0, false);
        check(
            st.kind == Start::kContinue && st.session == 0 && st.cells == 16000 && st.park_rest,
            "where the session it takes is the lender's (used longest ago), it continues in place, the lender parked");
        two[0].stamp = 3;   // A used last: B's session is the one to take
        st = choose_start(two, 0, none, sibling, {}, 0, false);
        check(st.kind == Start::kBorrowSession && st.from == 0 && st.session == 1,
              "... and from the other session, copied, where that one is");
        st = choose_start(two, 1, none, sibling, {}, 0, false);
        check(st.kind == Start::kContinue && st.session == 1 && st.cells == 16000,
              "a session that decodes is never the one a request takes: the sibling goes into the other, in place");
        st = choose_start(two, 2, none, next_of(two[1], 300), {}, 0, false);
        check(st.kind == Start::kBorrowSession && st.from == 1 && st.session == 0 && st.cells == 20000,
              "a decoding session's conversation lends its checkpoints, never its state");
        st = choose_start(two, 3, none, stranger, {}, 0, false);
        check(st.session < 0, "every session decoding: no place for a request");
        std::vector<Seq> tagged = seqs(talk(1000, 1), Seq{}, true);
        tagged[0].tag = 5;
        st = choose_start(tagged, 0, none, next_of(tagged[0], 300), {}, 0, false);
        check(st.kind == Start::kFresh && st.session == 1, "a conversation of another tag is not continued");
        Seq ended = conv(20000, {4000, 16000});   // a prompt of 20,000 cells whose end is kept with the engine's output
        ended.ckpts.push_back({20000, nullptr, true});
        for (int i = 0; i < 3000; ++i) ended.tokens.push_back(100 + i);
        std::vector<Seq> again = seqs(std::move(ended), Seq{}, true);
        st = choose_start(again, 0, none, prompt(20000), {}, 0, true);
        check(st.kind == Start::kContinue && st.session == 0 && st.cells == 20000 && !st.park_rest,
              "the same request again continues its conversation from the prompt's end, nothing parked");
        st = choose_start(again, 1, none, prompt(20000), {}, 0, true);
        check(st.kind == Start::kBorrowSession && st.from == 0 && st.session == 1 && st.cells == 20000,
              "... and beside the first one still decoding, from that end in a session of its own");
        // a session's conversation against a parked one with the same start: the session's goes first
        st = choose_start(gap, 0, parked, sibling, {}, 0, false);
        check(st.kind == Start::kBorrowSession && st.from == 0, "of equal starts a session's lends before the park's");
    }
    {                                                                                // 7. the disk tier
        auto filed = [](int64_t cells, std::vector<int64_t> pos, uint64_t bytes) {   // parked in a file of `bytes`
            Seq s = conv(cells, std::move(pos));
            s.disk_bytes = bytes;
            return s;
        };
        std::vector<Seq> evicted;
        ConversationCache c(40000, 2, 0, kCkpt);
        c.park(conv(1000, {1000}));   // RAM: 10,100
        c.park(filed(3000, {3000}, 50000));
        c.park(filed(4000, {4000}, 60000));
        check(c.size() == 3 && c.bytes() == 10100 && c.disk_size() == 2 && c.disk_bytes() == 110000,
              "files take no RAM: bytes() counts the RAM, disk_bytes() the files");
        check(c.make_room(10000, evicted) && evicted.empty(), "nor slots: two of them, one conversation in RAM");
        check(c.make_room(35000, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 1000 && c.size() == 2,
              "the RAM's evictions take conversations in RAM only");
        if (evicted.size() != 1) return 1;
        const size_t gone = c.evictions();
        Seq moved = std::move(evicted[0]);
        moved.disk_bytes = 12000;   // written to a file
        const uint64_t stamp = moved.stamp;
        c.readmit(std::move(moved));
        check(c.size() == 3 && c.evictions() == gone - 1 && c.disk_size() == 3 && c.bytes() == 0,
              "one readmitted in a file is parked again, not counted gone, and takes no RAM");
        check(c.entries()[0].stamp == stamp && c.entries()[0].kv_cells == 1000,
              "with its use as it was: still the least recently used");
        evicted.clear();
        check(c.make_disk_room(30000, 150000, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 1000 &&
                  c.disk_bytes() == 110000 && c.evictions() == gone,
              "a file that does not fit evicts the least recently used file, counted gone");
        c.entries()[0].pin = 3000;   // the 3,000-cell one lends its start
        evicted.clear();
        check(c.make_disk_room(80000, 150000, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 4000,
              "a lending conversation's file is never evicted: the next least recently used goes");
        evicted.clear();
        check(!c.make_disk_room(120000, 150000, evicted) && evicted.empty() && c.size() == 1,
              "nothing is evicted when the file would not fit even so");
        check(!c.make_disk_room(200000, 150000, evicted) && evicted.empty(), "nor for one over the whole limit");
        ConversationCache s(1u << 30, 1, 0, kCkpt);
        s.park(filed(3000, {3000}, 50000));
        s.park(conv(1000, {1000}));
        evicted.clear();
        check(s.make_room(100, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 1000 && s.disk_size() == 1,
              "one slot: the conversation in RAM gives way, the file stays");
        s.park(conv(2000, {2000}));
        evicted.clear();
        check(s.shed(-1000000, evicted) == 1 && evicted.size() == 1 && evicted[0].kv_cells == 2000 && s.size() == 1 &&
                  s.disk_size() == 1,
              "shed() gives the RAM back: files stay");
        check(s.min_tokens() == 0 && ConversationCache(0, 0, 512, kCkpt).min_tokens() == 512,
              "the shortest conversation the park keeps, for the disk tier too");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
