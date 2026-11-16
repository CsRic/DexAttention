#pragma once
// Host side of the shared runtime: the COPY-ENGINE executor of a message table.
//
// The intra-node messages of a call can go by copy engine instead of block 0's engine: the host enqueues every one of
// them, right after the kernel launch, on side streams; a message whose trigger is not "start" first waits
// (cuStreamWaitValue64) for its trigger word to reach the call's epoch:
//   kind 0  start          no wait
//   kind 1  arrival        the signal word `word` of this PE (an input landed here: relay forwards)
//   kind 2  completion     the trigger flag `word` (set by the kernel when the slot / row it covers is complete)
// Streams: start and completion messages on one stream per destination PE (in the order given: start messages first,
// then completion messages in expected completion order); arrival messages on ONE forward stream (expected
// arrival order); each stream is enqueued by its own host thread. The signal of a message is an 8-byte copy of the call's epoch into the destination's signal word, stream-
// ordered after its bytes. Credits (pairwise, optional): a stream waits for its destination's credit before its first
// copy. At the end every side stream is joined into `main`, and (optional) the forward stream stores the epoch into
// fw_done (the kernel's last block waits for it before releasing credits).
#include <cstdint>
#include <algorithm>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace rt {

// host table, int64 [n, 6] in python; nsig consecutive signal words sig .. sig + nsig - 1 are set after the bytes (a run of
// consecutive slots moved by one copy)
struct CeMsg { int64_t src, dst, bytes; int32_t pe, sig; int32_t kind, word; int64_t nsig; };

struct CeExec {
    std::vector<cudaStream_t> peer;       // [P + 1]: start + completion messages to PE q on stream q; every arrival
                                          // (forward) message on stream P. (Each extra stream that blocks in a wait
                                          // needs its own hardware queue, so forwards share one stream.)
    std::vector<cudaEvent_t> done;        // per stream
    cudaEvent_t go = nullptr;
    uint64_t *epoch_dev = nullptr;        // kEpochs copies of the call's epoch (a run's signal words in one copy)
    static constexpr int kEpochs = 1024;
    std::vector<uint64_t> epoch_host;

    void ensure(int P) {
        if ((int)peer.size() >= P) return;
        peer.resize(P + 1, nullptr); done.resize(P + 1, nullptr);
        for (auto &st : peer) cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        for (auto &e : done) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&go, cudaEventDisableTiming);
        cudaMalloc(&epoch_dev, kEpochs * sizeof(uint64_t));
        epoch_host.assign(kEpochs, 0);
    }

    // base: this PE's region; peer_base[pe] / peer_sig[pe]: the P2P-mapped region / signal words of each intra-node
    // peer; my_sig: this PE's signal words; flags: trigger flags (device, this PE). Call AFTER the kernel launch on
    // `main` (the copies queue behind nothing but `go`, recorded before the launch by begin()).
    void begin(cudaStream_t main, uint64_t epoch) {
        // pageable source: the copy is staged before cudaMemcpyAsync returns, so the buffer can be refilled next call
        std::fill(epoch_host.begin(), epoch_host.end(), epoch);
        cudaMemcpyAsync(epoch_dev, epoch_host.data(), kEpochs * sizeof(uint64_t), cudaMemcpyHostToDevice, main);
        cudaEventRecord(go, main);
    }
    // One host thread per stream enqueues that stream's messages: a stream holds only ~1K pending operations, and a
    // full one blocks the enqueuing thread until the GPU makes progress -- which can need messages of ANOTHER stream
    // that a single thread would not have enqueued yet (a cross-GPU wait cycle).
    void enqueue(cudaStream_t main, const CeMsg *m, int n, char *base, const std::vector<char *> &peer_base,
                 const std::vector<uint64_t *> &peer_sig, uint64_t *my_sig, unsigned long long *flags, uint64_t epoch,
                 const uint64_t *credit, uint64_t call_id, unsigned long long *fw_done) {
        const int P = (int)peer.size() - 1;
        std::vector<std::vector<int>> per(P + 1);                  // stream index: destination PE, P = forwards
        for (int i = 0; i < n; ++i) per[m[i].kind == 1 ? P : m[i].pe].push_back(i);
        int dev = 0; cudaGetDevice(&dev);
        auto run = [&, dev](int si) {
            cudaSetDevice(dev);
            cudaStream_t s = peer[si];
            cudaStreamWaitEvent(s, go, 0);
            std::vector<char> cred(P, 0);
            for (int i : per[si]) {
                const CeMsg &x = m[i];
                if (credit && !cred[x.pe]) {           // the destination finished the previous call
                    cuStreamWaitValue64(s, (CUdeviceptr)(credit + x.pe), (cuuint64_t)(call_id - 1), CU_STREAM_WAIT_VALUE_GEQ);
                    cred[x.pe] = 1;
                }
                if (x.kind == 1)
                    cuStreamWaitValue64(s, (CUdeviceptr)(my_sig + x.word), (cuuint64_t)epoch, CU_STREAM_WAIT_VALUE_GEQ);
                else if (x.kind == 2)
                    cuStreamWaitValue64(s, (CUdeviceptr)(flags + x.word), (cuuint64_t)epoch, CU_STREAM_WAIT_VALUE_GEQ);
                cudaMemcpyAsync(peer_base[x.pe] + x.dst, base + x.src, (size_t)x.bytes, cudaMemcpyDeviceToDevice, s);
                if (x.sig >= 0)
                    for (int64_t k = 0; k < x.nsig; k += kEpochs)
                        cudaMemcpyAsync(peer_sig[x.pe] + x.sig + k, epoch_dev, std::min<int64_t>(kEpochs, x.nsig - k) * 8,
                                        cudaMemcpyDeviceToDevice, s);
            }
            cudaEventRecord(done[si], s);
        };
        std::vector<std::thread> th;
        for (int si = 0; si <= P; ++si) if (!per[si].empty()) th.emplace_back(run, si);
        for (auto &t : th) t.join();
        if (fw_done) {                                     // every forward done -> the kernel's last block may release
            if (per[P].empty()) cudaStreamWaitEvent(peer[P], go, 0);                                    // its credits
            cudaMemcpyAsync(fw_done, epoch_dev, sizeof(uint64_t), cudaMemcpyDeviceToDevice, peer[P]);
            cudaEventRecord(done[P], peer[P]);
            cudaStreamWaitEvent(main, done[P], 0);
        }
        for (int si = 0; si <= P; ++si)
            if (!per[si].empty()) cudaStreamWaitEvent(main, done[si], 0);   // the next call's staging waits for every read
    }
};

}  // namespace rt
