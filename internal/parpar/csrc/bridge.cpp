// bridge.cpp - C bridge around PAR2ProcCPU for CGo import.
//
// USE_LIBUV is intentionally NOT defined; this uses the std::future code path
// which is self-contained and does not require libuv.

#include "bridge.h"
#include "vendor/gf16/controller_cpu.h"
#include "vendor/gf16/gfmat_coeff.h"

#include <mutex>
#include <new>

// One-time initialization of gfmat coefficient tables.
// gfmat_init() is normally called by PAR2Proc's constructor; since we bypass
// PAR2Proc and use PAR2ProcCPU directly, we call it here instead.
static std::once_flag s_gfmat_once;
static void ensure_gfmat_init() {
    std::call_once(s_gfmat_once, gfmat_init);
}

struct parpar_gfproc {
    PAR2ProcCPU* cpu;
    size_t        sliceSize;
};

extern "C" {

parpar_gfproc_t* parpar_gfproc_new(size_t sliceSize, int numThreads) {
    ensure_gfmat_init();

    parpar_gfproc_t* proc = new(std::nothrow) parpar_gfproc_t;
    if (!proc) return nullptr;

    proc->cpu = new(std::nothrow) PAR2ProcCPU(2); // 2 staging areas
    if (!proc->cpu) {
        delete proc;
        return nullptr;
    }

    proc->sliceSize = sliceSize;

    // Store custom thread count before init() so init() will use it when
    // setting up the worker pool (setNumThreads called inside init()).
    if (numThreads > 0) {
        proc->cpu->setNumThreads(numThreads);
    }

    // setSliceSize must be called before init().
    proc->cpu->setSliceSize(sliceSize);

    // init() creates the GF16 multiplier, allocates staging buffers, starts
    // the transfer thread, and launches compute worker threads.
    if (!proc->cpu->init(GF16_AUTO, 0, 0)) {
        delete proc->cpu;
        delete proc;
        return nullptr;
    }

    return proc;
}

void parpar_gfproc_free(parpar_gfproc_t* proc) {
    if (!proc) return;
    if (proc->cpu) {
        proc->cpu->deinit();
        delete proc->cpu;
    }
    delete proc;
}

void parpar_gfproc_set_recovery_slices(parpar_gfproc_t* proc, const uint16_t* exponents, unsigned count) {
    if (!proc || !proc->cpu) return;
    proc->cpu->setRecoverySlices(count, exponents);
}

void parpar_gfproc_set_num_threads(parpar_gfproc_t* proc, int n) {
    if (!proc || !proc->cpu) return;
    proc->cpu->setNumThreads(n);
}

int parpar_gfproc_add(parpar_gfproc_t* proc, unsigned sliceNum, const void* data, size_t len) {
    if (!proc || !proc->cpu) return PARPAR_ADD_OK;

    // Record the staging state before we potentially wait.
    PAR2ProcBackendAddResult result = proc->cpu->canAdd();

    // If all staging areas are busy, block until one becomes available.
    if (result == PROC_ADD_FULL) {
        proc->cpu->waitForAdd();
    }

    // Add the input slice; the returned future resolves when the prepare/
    // transfer phase is done (not when compute finishes — that is async).
    std::future<void> f = proc->cpu->addInput(data, len, (uint16_t)sliceNum, /*flush=*/false);
    f.get();

    return (int)result;
}

void parpar_gfproc_end(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return;
    proc->cpu->flush();  // flush inputs buffered but not yet submitted to compute workers
    // endInput() returns a future that resolves once all compute threads have finished.
    std::future<void> f = proc->cpu->endInput();
    f.get();
}

void parpar_gfproc_get_output(parpar_gfproc_t* proc, unsigned recoveryIdx, void* dst, size_t len) {
    if (!proc || !proc->cpu || !dst) return;
    (void)len; // informational; getOutput uses the internally-tracked slice size
    std::future<bool> f = proc->cpu->getOutput(recoveryIdx, dst);
    f.get();
}

void parpar_gfproc_free_mem(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return;
    proc->cpu->freeProcessingMem();
}

const char* parpar_gfproc_method_name(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return "unknown";
    return proc->cpu->getMethodName();
}

unsigned parpar_gfproc_num_threads(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return 0;
    return (unsigned)proc->cpu->getNumThreads();
}

} // extern "C"
