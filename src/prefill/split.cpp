// src/prefill/split.cpp - see include/lamina/prefill/split.hpp.
#include "lamina/prefill/split.hpp"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/geometry.hpp"
#include "lamina/core/layer.hpp"
#include "lamina/core/layout.hpp"
#include "lamina/core/native_head.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>

namespace lamina::prefill {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t SUB = 2048;   // the prompt path's sub-chunk (prefill.cpp): the residual goes across in these
uint64_t up(uint64_t n) { return (n + 255) & ~255ull; }
/// NativeDense's zeroed tail after each native matrix: the prompt path's MMQ reads past the last row of one whose rows
/// are not whole 256-value steps (the shared expert's down projection, 640)
constexpr uint64_t kNativeTail = 4096;

/// A copy of `bytes` from `src` to `dst`, each a device pointer on its own GPU.
struct Run {
    const uint8_t* src;
    uint8_t* dst;
    uint64_t bytes;
};

/// The runs that copy cells [c0, c1) of one K/V array, laid out [page][kv_head][page_size][row] in both states, from
/// one whose physical pages start at logical page `src_first` to one whose start at `dst_first`: whole pages in one
/// run, a page's part one run a head.
void cell_runs(int64_t c0, int64_t c1, int64_t ps, int64_t nkv, uint64_t rb, const void* src, int64_t src_first,
               void* dst, int64_t dst_first, std::vector<Run>& out) {
    const uint64_t page = (uint64_t) nkv * (uint64_t) ps * rb;
    auto at = [&](const void* base, int64_t first, int64_t p, uint64_t in) {
        return (uint8_t*) base + (uint64_t) (p - first) * page + in;
    };
    for (int64_t c = c0; c < c1;) {
        const int64_t p = c / ps;
        if (c == p * ps && c1 >= (p + 1) * ps) {
            const int64_t q = c1 / ps;   // pages [p, q) whole
            out.push_back({at(src, src_first, p, 0), at(dst, dst_first, p, 0), (uint64_t) (q - p) * page});
            c = q * ps;
            continue;
        }
        const int64_t e = std::min(c1, (p + 1) * ps);
        for (int64_t h = 0; h < nkv; ++h) {
            const uint64_t in = ((uint64_t) h * (uint64_t) ps + (uint64_t) (c - p * ps)) * rb;
            out.push_back({at(src, src_first, p, in), at(dst, dst_first, p, in), (uint64_t) (e - c) * rb});
        }
        c = e;
    }
}

/// A tensor of a stage's layers: the main GPU's row, and its buffers' bytes and offsets in the copies.
struct Tensor {
    std::string name;
    core::WeightRef ref;
    uint64_t off[4] = {}, bytes[4] = {};   // data, native_data, hc_codes, hc_scales
};

struct Job {
    bool chunk = false;   // false: the session state at the prompt's start, and the weights, in
    const int64_t* tokens = nullptr;
    int64_t T = 0, p0 = 0;
    bool run_end = false;
    std::vector<cudaEvent_t> landed;
    std::vector<std::pair<int64_t, void*>> marks;   // set_marks' in (p0, p0 + T]
};

}   // namespace

/// What the stages share.
struct PromptSplit::Impl {
    struct St;
    PromptSplit* self = nullptr;
    Prefill* p1 = nullptr;
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* geo = nullptr;
    core::SessionState* ss = nullptr;
    core::MtpDrafter* mtp = nullptr;
    int64_t max_chunk = 0;
    int main = 0;
    lamina::kernels::QsaShapes shapes;
    int64_t pos0 = 0, first_cell = 0;
    bool on = false, ckpts_on = false;
    std::vector<std::pair<int64_t, void*>> marks;   // set_marks: the checkpoints inside the first stage's run
    std::vector<std::unique_ptr<St>> st;            // the stages after the main GPU's, in the layers' order
    std::vector<int32_t> order;                     // set_order, for the stages init makes
    std::mutex stats_mu;                            // ms_copy_in, added to from the stages' threads
};

/// A stage after the main GPU's: its prompt path, its copies and its thread.  The stage before it (the main GPU's
/// prompt path, or the stage before on its own thread) hands it each chunk's residual through SplitOut.
struct PromptSplit::Impl::St final : public SplitOut {
    Impl* im = nullptr;
    bool last = false;   // the model's last layers: the draft layer's prompt K/V are its work
    int dev = -1;
    int64_t at = 0, end = 0;                       // its layers
    int64_t run_p0 = 0, run_T = 0;                 // the chunk its prompt path runs (PromptSplit::on_progress)
    std::unique_ptr<Prefill> own_p{new Prefill};   // (ends with its GPU current: ~St)
    Prefill& p = *own_p;
    // the dense weights of its layers and, the last stage, the draft layer's prompt-K/V weights: the main GPU's rows,
    // the copies' layout, and this GPU's table of its copies
    std::vector<Tensor> tensors;
    uint64_t dkv_off[10] = {}, dkv_bytes[10] = {};   // MtpPromptKv's weights, in its field order
    uint64_t wbytes = 0;
    struct WSrc {
        const void* dev;
        uint64_t off, bytes;
    };
    std::vector<WSrc> wsrc;   // the main GPU's buffers and their place in the copies
    core::WeightTable wt2;
    // its session state, indexed from `at`: its QSA layers' copies, its GDN layers' (the main session's GDN states
    // [gdn_first, gdn_first + gdn_count)), and the draft layer's cells of one chunk at a time
    core::SessionState ss2;
    std::vector<core::QsaState> qsa2;
    std::vector<int64_t> qsa_main;   // their index among the main session's QSA states
    int64_t gdn_first = 0, gdn_count = 0;
    core::QsaState draft2;
    core::MtpPromptKv dkv2;
    uint8_t *w_base = nullptr, *draft_base = nullptr;
    // its stage stream and copy stream; the main GPU's stream for the copies to and from its session
    cudaStream_t s2 = nullptr, cp2 = nullptr, back = nullptr;
    std::vector<cudaEvent_t> ev_in;   // its GPU: a piece of the residual is in the stage's buffers
    cudaEvent_t ev_free = nullptr;    // its GPU: the pinned residual has been read
    cudaEvent_t ev_a = nullptr;       // its GPU's last use of the staging buffer
    cudaEvent_t ev_m = nullptr;       // the main GPU's last use of the staging buffer
    float* host_R = nullptr;          // pinned, portable: [max_chunk, hc * n_embd]
    uint8_t* stage = nullptr;         // pinned, portable
    uint64_t stage_bytes = 0;
    // its thread and its queue
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    bool quit = false, busy = false, failed = false;
    std::string err;
    int64_t pushed = 0, taken = 0;   // chunks handed on, and those whose residual the stage has read
    int64_t ran_pos = -1;            // ran(): the main thread's checkpoint decisions up to there are made
    std::vector<std::pair<int64_t, void*>> ckpts;

    const core::QsaState& main_qsa(size_t i) const { return im->ss->qsa_states[qsa_main[i]]; }

    /// After its thread has ended: its prompt path and its streams with its GPU current, the main GPU's copy stream
    /// with that one; the main GPU is current afterwards.
    ~St() override {
        if (th.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu);
                quit = true;
            }
            cv.notify_all();
            th.join();
        }
        if (dev < 0) return;
        cudaSetDevice(dev);
        for (cudaStream_t x : {s2, cp2})
            if (x) cudaStreamSynchronize(x);
        own_p.reset();
        for (cudaStream_t x : {s2, cp2})
            if (x) cudaStreamDestroy(x);
        for (cudaEvent_t e : ev_in)
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : {ev_free, ev_a})
            if (e) cudaEventDestroy(e);
        cudaSetDevice(im->main);
        if (back) {
            cudaStreamSynchronize(back);
            cudaStreamDestroy(back);
        }
        if (ev_m) cudaEventDestroy(ev_m);
        if (host_R) cudaFreeHost(host_R);
        if (stage) cudaFreeHost(stage);
    }

    /// `runs` through the staging buffer, a buffer's worth at a time: from the main GPU's memory to this stage's
    /// (`in`) or back.  Each side's copies go out once the buffer is theirs, waited for on the host: a copy queued
    /// behind a wait holds up every copy queued after it on its GPU's copy engine (another stage's too).  The last
    /// batch's copies to the reading side are left running (`last_read`).  The caller may have either device current.
    cudaEvent_t last_read = nullptr;
    bool stage_copy(const std::vector<Run>& runs, bool in, std::string& e) {
        int cur = 0;
        cudaGetDevice(&cur);
        std::vector<Run> batch;
        uint64_t used = 0;
        auto flush = [&]() -> bool {
            if (batch.empty()) return true;
            // the side that writes the staging buffer, then the side that reads it
            const int wdev = in ? im->main : dev, rdev = in ? dev : im->main;
            cudaStream_t ws = in ? back : cp2, rs = in ? cp2 : back;
            cudaEvent_t wev = in ? ev_m : ev_a, rev = in ? ev_a : ev_m;
            if (last_read != nullptr && cudaEventSynchronize(last_read) != cudaSuccess) {
                e = std::string("prompt split: copies: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            cudaSetDevice(wdev);
            uint64_t o = 0;
            for (const Run& r : batch) {
                cudaMemcpyAsync(stage + o, r.src, (size_t) r.bytes, cudaMemcpyDeviceToHost, ws);
                o += up(r.bytes);
            }
            cudaEventRecord(wev, ws);
            if (cudaEventSynchronize(wev) != cudaSuccess) {
                e = std::string("prompt split: copies: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            cudaSetDevice(rdev);
            o = 0;
            for (const Run& r : batch) {
                cudaMemcpyAsync(r.dst, stage + o, (size_t) r.bytes, cudaMemcpyHostToDevice, rs);
                o += up(r.bytes);
            }
            cudaEventRecord(rev, rs);
            last_read = rev;
            batch.clear();
            used = 0;
            const cudaError_t x = cudaGetLastError();
            if (x != cudaSuccess) {
                e = std::string("prompt split: copies: ") + cudaGetErrorString(x);
                return false;
            }
            return true;
        };
        for (Run r : runs)
            while (r.bytes > 0) {
                if (used >= stage_bytes && !flush()) return false;
                const uint64_t n = std::min(r.bytes, stage_bytes - used);
                batch.push_back({r.src, r.dst, n});
                used += up(n);
                r.src += n;
                r.dst += n;
                r.bytes -= n;
            }
        const bool ok = flush();
        cudaSetDevice(cur);
        return ok;
    }

    /// The runs of cells [c0, c1) of every QSA layer here (and the pooled keys of their blocks), from this stage's
    /// copies to the main session's (`back_to_main`) or the other way.
    void qsa_runs(int64_t c0, int64_t c1, bool back_to_main, std::vector<Run>& runs) const {
        if (c0 >= c1) return;
        const lamina::kernels::QsaShapes& s = im->shapes;
        const core::ModelGeometry& g = *im->geo;
        const uint64_t row = (uint64_t) s.idx_dim * 4;
        const int64_t r0 = c0 / s.idx_block, r1 = c1 / s.idx_block + 1;   // the open block's row too
        for (size_t i = 0; i < qsa2.size(); ++i) {
            const auto a2 = core::qsa_kv_arrays(qsa2[i], g), am = core::qsa_kv_arrays(main_qsa(i), g);
            for (size_t k = 0; k < a2.size(); ++k)
                if (back_to_main)
                    cell_runs(c0, c1, s.page_size, s.n_head_kv, a2[k].row_bytes, a2[k].data, 0, am[k].data, 0, runs);
                else cell_runs(c0, c1, s.page_size, s.n_head_kv, am[k].row_bytes, am[k].data, 0, a2[k].data, 0, runs);
            const uint8_t* px = (const uint8_t*) qsa2[i].idx_pooled + (uint64_t) r0 * row;
            const uint8_t* pm = (const uint8_t*) main_qsa(i).idx_pooled + (uint64_t) r0 * row;
            const uint64_t n = (uint64_t) (r1 - r0) * row;
            runs.push_back(back_to_main ? Run{px, (uint8_t*) pm, n} : Run{pm, (uint8_t*) px, n});
        }
    }

    /// The fixed-size state of the layers here: the GDN states and the indexers' tails, spare keys and block positions.
    void state_runs(bool back_to_main, std::vector<Run>& runs) const {
        const lamina::kernels::QsaShapes& s = im->shapes;
        const core::ModelGeometry& g = *im->geo;
        const uint64_t row = (uint64_t) s.idx_dim * 4;
        const uint64_t gdn = (uint64_t) gdn_count * core::gdn_state_floats(g) * 4;
        uint8_t* gm = (uint8_t*) (im->ss->gdn_state + (uint64_t) gdn_first * core::gdn_state_floats(g));
        auto add = [&](const void* x2, const void* xm, uint64_t n) {
            if (n == 0) return;
            runs.push_back(back_to_main ? Run{(const uint8_t*) x2, (uint8_t*) xm, n}
                                        : Run{(const uint8_t*) xm, (uint8_t*) x2, n});
        };
        add(ss2.gdn_state, gm, gdn);
        for (size_t i = 0; i < qsa2.size(); ++i) {
            add(qsa2[i].idx_tail, main_qsa(i).idx_tail, (uint64_t) (s.idx_block - 1) * row);
            add(qsa2[i].idx_dead, main_qsa(i).idx_dead, row);
            add(qsa2[i].idx_block_pos, main_qsa(i).idx_block_pos, 4);
        }
    }

    /// The draft layer's cells of one chunk: its copy for cells [p0, p0 + T).
    uint64_t draft_bytes(int64_t chunk, int64_t cells) const {
        const lamina::kernels::QsaShapes& s = im->shapes;
        const int64_t pages = (cells + s.page_size - 1) / s.page_size,
                      held = (chunk + s.page_size - 1) / s.page_size + 1;
        return core::qsa_state_prompt_bytes(*im->geo, cells, std::max<int64_t>(pages - held, 0), false,
                                            core::qsa_draft_kv_format());
    }

    uint64_t bytes_for(int64_t chunk, int64_t cells) const {
        uint64_t n = up(p.bytes_for(chunk)) + up(wbytes);
        n += (uint64_t) qsa2.size() * up(core::qsa_state_prompt_bytes(*im->geo, cells, 0, true, core::qsa_kv_format()));
        n += up((uint64_t) gdn_count * core::gdn_state_floats(*im->geo) * 4);
        if (last && im->mtp != nullptr) n += up(draft_bytes(chunk, cells));
        return n;
    }

    // ---- its thread
    bool copy_in(std::string& e) {
        const auto t0 = Clock::now();
        // the weights from the main GPU's, through the staging buffer beside the first stage's first chunk (a pinned
        // host copy of them was no faster: 0.5% of a 32K-token prompt)
        std::vector<Run> runs;
        for (const WSrc& w : wsrc) runs.push_back({(const uint8_t*) w.dev, w_base + w.off, w.bytes});
        state_runs(false, runs);
        qsa_runs(0, im->pos0, false, runs);
        if (!stage_copy(runs, true, e)) return false;
        cudaStreamWaitEvent(s2, ev_a, 0);   // the stage's work after its state and weights
        cudaEvent_t w_done = nullptr;
        cudaEventCreateWithFlags(&w_done, cudaEventDisableTiming);
        cudaEventRecord(w_done, cp2);
        cudaStreamWaitEvent(s2, w_done, 0);
        cudaEventDestroy(w_done);
        if (cudaStreamSynchronize(cp2) != cudaSuccess) {
            e = std::string("prompt split: copy in: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        {
            std::lock_guard<std::mutex> lk(im->stats_mu);
            im->self->ms_copy_in += core::ms_since(t0);
        }
        return true;
    }

    bool run_chunk(const Job& j, std::string& e) {
        const core::ModelGeometry& g = *im->geo;
        const lamina::kernels::QsaShapes& s = im->shapes;
        const int64_t D = g.hc * g.n_embd, n = (j.T + SUB - 1) / SUB;
        // the residual's pieces, each queued once it has landed in host memory (a copy queued behind a wait would hold
        // up another stage's copies on this GPU's engine) and, a stage that hands its own on, once the last chunk's
        // rows in its place have gone out
        float* R = p.residual();
        for (int64_t k = 0; k < n; ++k) {
            const int64_t P = std::min(SUB, j.T - k * SUB);
            const cudaEvent_t gone = p.out_done(k);
            if (cudaEventSynchronize(j.landed[(size_t) k]) != cudaSuccess ||
                (gone != nullptr && cudaEventSynchronize(gone) != cudaSuccess)) {
                e = std::string("prompt split: the residual: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            cudaMemcpyAsync(R + k * SUB * D, host_R + k * SUB * D, (size_t) (P * D * 4), cudaMemcpyHostToDevice, cp2);
            cudaEventRecord(ev_in[(size_t) k], cp2);
        }
        cudaEventRecord(ev_free, cp2);
        {
            std::lock_guard<std::mutex> lk(mu);
            ++taken;
        }
        cv.notify_all();
        // the last chunk's cells have gone out (its copies read the copies this chunk writes next to them)
        cudaStreamWaitEvent(s2, ev_a, 0);
        const bool drafts = last && im->mtp != nullptr;
        if (drafts && core::qsa_state_prompt_init(g, j.p0 + j.T, j.p0 / s.page_size, false, core::qsa_draft_kv_format(),
                                                  draft_base, draft2, s2) == 0) {
            e = "prompt split: the draft layer's cells";
            return false;
        }
        dkv2.kv = &draft2;
        dkv2.first_cell = im->first_cell;
        p.set_in_events(ev_in.data(), n);
        p.set_run_end(j.run_end);
        run_p0 = j.p0;
        run_T = j.T;
        // the first stage's checkpoints inside its run: those inside this chunk as this prompt path's marks, one at
        // the chunk's end (where the first stage's next chunk starts) after it
        std::vector<std::pair<int64_t, void*>> inside, at_end;
        for (const auto& k : j.marks) (k.first < j.p0 + j.T ? inside : at_end).push_back(k);
        p.set_marks(inside);   // (set_marks reserved its buffers)
        if (!p.run(j.tokens, j.T, j.p0, e)) {
            p.checkpoint_wait();
            return false;
        }
        if (!p.checkpoint_wait()) {
            e = "prompt split: a stage's part of a checkpoint";
            return false;
        }
        for (const auto& [pos, host] : at_end)
            if (!core::session_ckpt_save_layers(ss2, g, pos, host, s2, at, end, at)) {
                e = "prompt split: a stage's part of a checkpoint";
                return false;
            }
        // the chunk's cells to the main GPU's session
        std::vector<Run> runs;
        qsa_runs(j.p0, j.p0 + j.T, true, runs);
        if (drafts) {
            const core::QsaState& dm = im->mtp->kv_state();
            const auto a2 = core::qsa_kv_arrays(draft2, g), am = core::qsa_kv_arrays(dm, g);
            const int64_t c0 = std::max(im->first_cell, j.p0), fp = j.p0 / s.page_size;
            for (size_t k = 0; k < a2.size() && c0 < j.p0 + j.T; ++k)
                cell_runs(c0, j.p0 + j.T, s.page_size, s.n_head_kv, a2[k].row_bytes, a2[k].data, fp, am[k].data, 0,
                          runs);
        }
        if (!stage_copy(runs, false, e)) return false;
        // a prompt-cache checkpoint at its end: its layers' part, once the main thread has decided
        if (j.run_end && im->ckpts_on) {
            const int64_t pos = j.p0 + j.T;
            void* host = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || ran_pos >= pos; });
                for (size_t i = 0; i < ckpts.size(); ++i)
                    if (ckpts[i].first == pos) {
                        host = ckpts[i].second;
                        ckpts.erase(ckpts.begin() + (ptrdiff_t) i);
                        break;
                    }
            }
            if (host != nullptr && !core::session_ckpt_save_layers(ss2, g, pos, host, s2, at, end, at)) {
                e = "prompt split: a stage's part of the checkpoint";
                return false;
            }
        }
        return true;
    }

    void worker() {
        cudaSetDevice(dev);
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !jobs.empty(); });
                if (jobs.empty()) return;
                j = std::move(jobs.front());
                jobs.pop_front();
                busy = true;
            }
            std::string e;
            bool ok = failed;
            try {   // (a kernel's wrapper that throws would otherwise end the process without a word)
                ok = ok || (j.chunk ? run_chunk(j, e) : copy_in(e));
            } catch (const std::exception& x) {
                e = x.what();
                ok = false;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                if (j.chunk && taken < pushed && failed) ++taken;   // a failed stage never read it
                if (!ok && !failed) {
                    failed = true;
                    err = e;
                    if (j.chunk && taken < pushed) ++taken;
                }
                busy = false;
            }
            cv.notify_all();
        }
    }

    // ---- the stage before it hands a chunk on (its thread)
    float* out_begin(std::string& e) override {
        const auto t0 = Clock::now();
        bool any = false;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return failed || taken >= pushed; });
            if (failed) {
                e = "prompt split: " + err;
                return nullptr;
            }
            any = pushed > 0;
        }
        // the last chunk's copies out of it are done (waited for here: the copies into it would hold up this GPU's
        // copies queued after them)
        if (any && cudaEventSynchronize(ev_free) != cudaSuccess) {
            e = std::string("prompt split: ") + cudaGetErrorString(cudaGetLastError());
            return nullptr;
        }
        if (this == im->st.front().get()) im->self->ms_first_waited += core::ms_since(t0);
        return host_R;
    }

    bool out_end(const int64_t* tokens, int64_t T, int64_t pos0, const cudaEvent_t* landed, int64_t piece, bool run_end,
                 std::string& e) override {
        if (piece != SUB || T > im->max_chunk) {
            e = "prompt split: a chunk out of range";
            return false;
        }
        Job j;
        j.chunk = true;
        j.tokens = tokens;
        j.T = T;
        j.p0 = pos0;
        j.run_end = run_end;
        j.landed.assign(landed, landed + (T + SUB - 1) / SUB);
        for (const auto& k : im->marks)
            if (k.first > pos0 && k.first <= pos0 + T) j.marks.push_back(k);
        {
            std::lock_guard<std::mutex> lk(mu);
            if (failed) {
                e = "prompt split: " + err;
                return false;
            }
            jobs.push_back(std::move(j));
            ++pushed;
        }
        cv.notify_all();
        return true;
    }
};

PromptSplit::PromptSplit() : impl_(new Impl) { impl_->self = this; }

PromptSplit::~PromptSplit() {
    Impl& m = *impl_;
    if (m.st.empty()) return;
    int cur = 0;
    cudaGetDevice(&cur);
    // in the layers' order: a stage's thread ends before the stage it hands chunks to, whose pointer its prompt path
    // holds until then
    for (auto& sp : m.st) {
        Impl::St& t = *sp;
        {
            std::lock_guard<std::mutex> lk(t.mu);
            t.quit = true;
        }
        t.cv.notify_all();
        if (t.th.joinable()) t.th.join();
    }
    m.st.clear();
    cudaSetDevice(cur);
}

bool PromptSplit::init(Prefill& first, const core::WeightTable& wt, const core::ModelGeometry& g,
                       core::SessionState& ss, core::MtpDrafter* mtp, const std::vector<Stage>& stages, int main_device,
                       core::ExpertSource* src, int64_t max_chunk, std::string& err) {
    Impl& m = *impl_;
    if (stages.empty() || stages.size() > 2) {
        err = "--prompt-split: one or two stages after the main GPU's";
        return false;
    }
    for (size_t i = 0; i < stages.size(); ++i)
        if (stages[i].first < 2 || stages[i].first >= g.n_layers || (i > 0 && stages[i].first <= stages[i - 1].first)) {
            err = "--prompt-split: each stage's first layer must be 2.." + std::to_string(g.n_layers - 1) +
                  ", above the stage before it";
            return false;
        }
    // the last stage gathers the draft layer's token rows on its own GPU: from VRAM it could not
    if (mtp != nullptr && (core::native_embed() == nullptr || !core::native_embed()->mapped())) {
        err = "--prompt-split: the draft layer's prompt K/V need the token embedding in mapped host memory";
        return false;
    }
    m.p1 = &first;
    m.wt = &wt;
    m.geo = &g;
    m.ss = &ss;
    m.mtp = mtp;
    m.main = main_device;
    m.max_chunk = max_chunk;
    m.shapes = core::qsa_shapes(g);
    const int64_t D = g.hc * g.n_embd;
    for (size_t i = 0; i < stages.size(); ++i) {
        m.st.push_back(std::make_unique<Impl::St>());
        Impl::St& t = *m.st.back();
        t.im = &m;
        t.last = i + 1 == stages.size();
        t.dev = stages[i].device;
        t.at = stages[i].first;
        t.end = t.last ? g.n_layers : stages[i + 1].first;
    }
    for (size_t i = 0; i < stages.size(); ++i) {
        Impl::St& t = *m.st[i];
        const bool drafts = t.last && mtp != nullptr;
        // ---- the tensors of its layers, and (the last stage) the draft layer's prompt-K/V weights
        for (const auto& [name, ref] : wt.all()) {
            if (name.rfind("blk.", 0) != 0) continue;
            const int64_t l = std::atoll(name.c_str() + 4);
            if (l < t.at || l >= t.end) continue;
            Tensor x;
            x.name = name;
            x.ref = ref;
            x.bytes[0] = ref.data != nullptr ? ref.bytes : 0;
            if (ref.native_data != nullptr) {
                int be = 0, bb = 0;
                if (!lamina::block_geometry((uint32_t) ref.native_type, be, bb) || be <= 0 ||
                    (ref.ne0 * ref.ne1) % be != 0) {
                    err = "--prompt-split: " + name + " has a native type without a block geometry";
                    return false;
                }
                x.bytes[1] = (uint64_t) (ref.ne0 * ref.ne1 / be) * (uint64_t) bb + kNativeTail;   // its tail copied too
            }
            if (ref.hc_codes != nullptr) x.bytes[2] = (uint64_t) (ref.ne0 * ref.ne1);
            if (ref.hc_scales != nullptr) x.bytes[3] = (uint64_t) (ref.ne0 * ref.ne1 / 32) * 2;
            for (int k = 0; k < 4; ++k) {
                x.off[k] = t.wbytes;
                t.wbytes += up(x.bytes[k]);
            }
            t.tensors.push_back(std::move(x));
        }
        const void* dkv_src[10] = {};
        if (drafts) {
            const core::MtpPromptKv k = mtp->prompt_kv();
            const void* f[10] = {k.norm_emb, k.norm_hidden, k.fc_emb, k.fc_hidden, k.hc_norm,
                                 k.hc_down,  k.hc_up,       k.k_proj, k.v_proj,    k.k_norm};
            for (int q = 0; q < 10; ++q) {
                dkv_src[q] = f[q];
                t.dkv_bytes[q] = f[q] != nullptr ? mtp->tensor_bytes(f[q]) : 0;
                if (t.dkv_bytes[q] == 0) {
                    err = "--prompt-split: a draft layer weight has no size";
                    return false;
                }
                t.dkv_off[q] = t.wbytes;
                t.wbytes += up(t.dkv_bytes[q]);
            }
        }
        weight_bytes += t.wbytes;
        for (const Tensor& x : t.tensors) {
            const void* d[4] = {x.ref.data, x.ref.native_data, x.ref.hc_codes, x.ref.hc_scales};
            for (int k = 0; k < 4; ++k)
                if (x.bytes[k] > 0) t.wsrc.push_back({d[k], x.off[k], x.bytes[k]});
        }
        for (int q = 0; q < 10 && drafts; ++q) t.wsrc.push_back({dkv_src[q], t.dkv_off[q], t.dkv_bytes[q]});
        // ---- the layers' state indices
        for (int64_t l = 0, qi = 0, gi = 0; l < g.n_layers; ++l) {
            if (core::is_qsa_layer(g, l)) {
                if (l >= t.at && l < t.end) t.qsa_main.push_back(qi);
                ++qi;
            } else {
                if (l < t.at) t.gdn_first = gi + 1;
                else if (l < t.end) ++t.gdn_count;
                ++gi;
            }
        }
        if (t.qsa_main.empty()) {   // (its prompt path sizes its selection buffers
                                    // by its first attention layer's state)
            err = "--prompt-split: layers " + std::to_string(t.at) + "-" + std::to_string(t.end - 1) +
                  " hold no attention layer: a stage needs one";
            return false;
        }
        // ---- its GPU's side
        cudaSetDevice(t.dev);
        int least = 0, greatest = 0;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        t.ev_in.assign((size_t) ((max_chunk + SUB - 1) / SUB), nullptr);
        bool ok = cudaStreamCreateWithPriority(&t.s2, cudaStreamNonBlocking, least) == cudaSuccess &&
                  cudaStreamCreateWithFlags(&t.cp2, cudaStreamNonBlocking) == cudaSuccess &&
                  cudaEventCreateWithFlags(&t.ev_free, cudaEventDisableTiming) == cudaSuccess &&
                  cudaEventCreateWithFlags(&t.ev_a, cudaEventDisableTiming) == cudaSuccess;
        for (cudaEvent_t& e : t.ev_in) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        // a chunk's cells of every K/V state here and the pooled keys of their blocks, through one staging buffer
        {
            uint64_t per_cell =
                (uint64_t) t.qsa_main.size() * (core::qsa_kv_cell_bytes(g, core::qsa_kv_format()) +
                                                (uint64_t) m.shapes.idx_dim * 4 / (uint64_t) m.shapes.idx_block);
            if (drafts) per_cell += core::qsa_kv_cell_bytes(g, core::qsa_draft_kv_format());
            t.stage_bytes = std::max<uint64_t>(
                64ull << 20, up((uint64_t) (max_chunk + 4 * m.shapes.page_size) * per_cell + (4ull << 20)));
        }
        ok = ok &&
             cudaHostAlloc((void**) &t.host_R, (size_t) max_chunk * (size_t) D * 4, cudaHostAllocPortable) ==
                 cudaSuccess &&
             cudaHostAlloc((void**) &t.stage, (size_t) t.stage_bytes, cudaHostAllocPortable) == cudaSuccess;
        if (!ok) {
            (void) cudaGetLastError();
            cudaSetDevice(main_device);
            err = "--prompt-split: a stage's streams, events or pinned buffers";
            return false;
        }
        // its session state: the QSA states sized as the main session's until `begin` makes the copies
        t.qsa2.assign(t.qsa_main.size(), core::QsaState{});
        for (core::QsaState& q : t.qsa2) q.max_cells = ss.qsa_states[0].max_cells;
        t.ss2.qsa_states = t.qsa2.data();
        t.ss2.k = ss.k;
        t.ss2.max_cells = ss.max_cells;
        t.p.set_layers(t.at, t.last ? -1 : t.end, t.last ? nullptr : m.st[i + 1].get());
        if (stages[i].offload == nullptr) {
            t.p.set_fast_link(true);
            // its GPU's queue of this stage's work short: another stage's shares queued after it wait for all of it
            // (1, 2 and 4 sections measured alike)
            t.p.set_pacing(2);
        }
        if (t.last)
            t.p.on_progress = [this, tp = &t](double done) {
                if (on_progress) on_progress(tp->run_p0, tp->run_T, done);
            };
        t.p.draft = drafts ? mtp : nullptr;
        t.p.draft_kv = drafts ? &t.dkv2 : nullptr;
        if (!m.order.empty()) t.p.set_order(m.order);
        if (!t.p.init(t.wt2, g, t.ss2, src, stages[i].cache, stages[i].res, max_chunk, t.s2, stages[i].offload, err)) {
            cudaSetDevice(main_device);
            err = "--prompt-split: " + err;
            return false;
        }
        cudaSetDevice(main_device);
        if (cudaStreamCreateWithFlags(&t.back, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&t.ev_m, cudaEventDisableTiming) != cudaSuccess) {
            err = "--prompt-split: the main GPU's copy stream";
            return false;
        }
    }
    for (auto& sp : m.st) {
        Impl::St* t = sp.get();
        t->th = std::thread([t] { t->worker(); });
    }
    return true;
}

int PromptSplit::stages() const { return (int) impl_->st.size(); }
int64_t PromptSplit::at(int stage) const { return impl_->st[(size_t) stage]->at; }
int PromptSplit::device(int stage) const { return impl_->st[(size_t) stage]->dev; }
void PromptSplit::set_order(const std::vector<int32_t>& order) {
    impl_->order = order;
    for (auto& t : impl_->st) t->p.set_order(order);
}
bool PromptSplit::active() const { return impl_->on; }
const PrefillStats& PromptSplit::stats(int stage) const { return impl_->st[(size_t) stage]->p.stats(); }
void PromptSplit::set_profile(bool on) {
    for (auto& t : impl_->st) t->p.profile = on;
}

uint64_t PromptSplit::bytes_for(int stage, int64_t chunk, int64_t cells) const {
    return impl_->st[(size_t) stage]->bytes_for(chunk, cells);
}

bool PromptSplit::begin(const std::vector<Region>& regions, int64_t chunk, int64_t pos0, int64_t cells,
                        bool checkpoints, std::string& err) {
    Impl& m = *impl_;
    if (m.on) {
        err = "prompt split: begin twice";
        return false;
    }
    if (regions.size() != m.st.size()) {
        err = "prompt split: a region for each stage";
        return false;
    }
    int cur = 0;
    cudaGetDevice(&cur);
    if (m.mtp != nullptr) m.first_cell = m.mtp->prompt_kv().first_cell;
    for (size_t i = 0; i < m.st.size(); ++i) {
        Impl::St& t = *m.st[i];
        // the copies after the stage's buffers, whose copies ahead take what is left: bytes_for sized that for the
        // residency before the loan, which made more of its experts stream
        const uint64_t fixed = t.bytes_for(chunk, cells) - up(t.p.bytes_for(chunk));
        if (regions[i].base == nullptr || regions[i].bytes < fixed + (64ull << 20)) {
            cudaSetDevice(cur);
            err = "prompt split: a stage's region is too small";
            return false;
        }
        cudaSetDevice(t.dev);
        uint8_t* p = (uint8_t*) regions[i].base;
        const uint64_t pb = (regions[i].bytes - fixed) & ~255ull;
        if (!t.p.bind(p, pb, chunk, err)) {
            cudaSetDevice(cur);
            return false;
        }
        p += pb;
        t.w_base = p;
        p += up(t.wbytes);
        // the weights' rows on its GPU
        for (const Tensor& x : t.tensors) {
            core::WeightRef r = x.ref;
            r.data = x.bytes[0] ? t.w_base + x.off[0] : nullptr;
            r.native_data = x.bytes[1] ? t.w_base + x.off[1] : nullptr;
            r.hc_codes = x.bytes[2] ? (const int8_t*) (t.w_base + x.off[2]) : nullptr;
            r.hc_scales = x.bytes[3] ? (const uint16_t*) (t.w_base + x.off[3]) : nullptr;
            r.native_q8_1 = nullptr;
            t.wt2.put(x.name, r);
        }
        if (t.last && m.mtp != nullptr) {
            const uint8_t* w = t.w_base;
            t.dkv2.norm_emb = (const float*) (w + t.dkv_off[0]);
            t.dkv2.norm_hidden = (const float*) (w + t.dkv_off[1]);
            t.dkv2.fc_emb = w + t.dkv_off[2];
            t.dkv2.fc_hidden = w + t.dkv_off[3];
            t.dkv2.hc_norm = (const float*) (w + t.dkv_off[4]);
            t.dkv2.hc_down = (const uint16_t*) (w + t.dkv_off[5]);
            t.dkv2.hc_up = (const uint16_t*) (w + t.dkv_off[6]);
            t.dkv2.k_proj = w + t.dkv_off[7];
            t.dkv2.v_proj = w + t.dkv_off[8];
            t.dkv2.k_norm = (const float*) (w + t.dkv_off[9]);
        }
        // its session state's copies, zeroed (the prefix's cells and the state at pos0 come in on its thread)
        for (core::QsaState& q : t.qsa2) {
            const uint64_t n = core::qsa_state_prompt_init(*m.geo, cells, 0, true, core::qsa_kv_format(), p, q, t.cp2);
            if (n == 0) {
                cudaSetDevice(cur);
                err = "prompt split: the K/V copies";
                return false;
            }
            p += up(n);
        }
        t.ss2.gdn_state = (float*) p;
        p += up((uint64_t) t.gdn_count * core::gdn_state_floats(*m.geo) * 4);
        t.draft_base = p;
    }
    cudaSetDevice(cur);
    m.pos0 = pos0;
    m.ckpts_on = checkpoints;
    m.marks.clear();
    m.on = true;
    for (auto& sp : m.st) {
        Impl::St& t = *sp;
        {
            std::lock_guard<std::mutex> lk(t.mu);
            t.pushed = t.taken = 0;
            t.ran_pos = -1;
            t.ckpts.clear();
            t.failed = false;
            t.err.clear();
            t.jobs.push_back(Job{});   // the state at pos0 and the weights, first
        }
        t.cv.notify_all();
    }
    m.p1->set_layers(0, m.st.front()->at, m.st.front().get());
    return true;
}

bool PromptSplit::set_marks(std::vector<std::pair<int64_t, void*>> marks) {
    Impl& m = *impl_;
    m.marks.clear();
    if (marks.empty()) return true;
    int cur = 0;
    cudaGetDevice(&cur);
    bool ok = true;
    for (auto& sp : m.st) {   // (each stage's buffers on its GPU, before it runs a chunk with them)
        cudaSetDevice(sp->dev);
        ok = ok && sp->p.reserve_marks();
    }
    cudaSetDevice(cur);
    if (ok) m.marks = std::move(marks);
    return ok;
}

void PromptSplit::checkpoint(int64_t pos, void* host) {
    for (const auto& k : impl_->marks)   // (the stages save theirs as they pass it)
        if (k.first == pos) return;
    for (auto& sp : impl_->st) {
        Impl::St& t = *sp;
        {
            std::lock_guard<std::mutex> lk(t.mu);
            t.ckpts.emplace_back(pos, host);
        }
        t.cv.notify_all();
    }
}

void PromptSplit::ran(int64_t pos) {
    for (auto& sp : impl_->st) {
        Impl::St& t = *sp;
        {
            std::lock_guard<std::mutex> lk(t.mu);
            t.ran_pos = std::max(t.ran_pos, pos);
        }
        t.cv.notify_all();
    }
}

bool PromptSplit::finish(std::string& err) {
    Impl& m = *impl_;
    if (!m.on) return true;
    const auto t0 = Clock::now();
    bool ok = true;
    for (auto& sp : m.st) {   // in the layers' order: a stage is done once the
                              // one before it has handed its last chunk on
        Impl::St& t = *sp;
        std::unique_lock<std::mutex> lk(t.mu);
        t.ran_pos = INT64_MAX;
        t.cv.notify_all();
        t.cv.wait(lk, [&] { return t.jobs.empty() && !t.busy; });
        if (t.failed && ok) {
            ok = false;
            err = "prompt split: " + t.err;
        }
    }
    m.on = false;
    m.p1->set_layers(0, -1, nullptr);
    int cur = 0;
    cudaGetDevice(&cur);
    for (auto& sp : m.st) {
        Impl::St& t = *sp;
        // the GDN states and the indexers' fixed-size state go back (the cells went after each chunk)
        if (ok) {
            std::vector<Run> runs;
            t.state_runs(true, runs);
            ok = t.stage_copy(runs, false, err);
        }
        cudaSetDevice(m.main);
        if (cudaStreamSynchronize(t.back) != cudaSuccess && ok) {
            err = std::string("prompt split: ") + cudaGetErrorString(cudaGetLastError());
            ok = false;
        }
        cudaSetDevice(t.dev);
        if (cudaStreamSynchronize(t.cp2) != cudaSuccess && ok) {
            err = std::string("prompt split: ") + cudaGetErrorString(cudaGetLastError());
            ok = false;
        }
        m.p1->add_routed(t.p.routed());
        t.p.clear_routed();
    }
    cudaSetDevice(cur);
    ms_finish += core::ms_since(t0);
    return ok;
}

}   // namespace lamina::prefill
