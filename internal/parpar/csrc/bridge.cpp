// bridge.cpp - C bridge around PAR2ProcCPU for CGo import.
//
// USE_LIBUV is intentionally NOT defined; this uses the std::future code path
// which is self-contained and does not require libuv.

#include "bridge.h"
#include "vendor/gf16/controller_cpu.h"
#include "vendor/gf16/gfmat_coeff.h"

#include <algorithm>
#include <mutex>
#include <new>
#include <vector>

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

// The Affine (GFNI+AVX512) kernel in the prebuilt libraries segfaults inside
// the compute workers on CPUs that expose GFNI together with AVX-512BW/VL
// (seen on Intel Xeon Platinum 8573C and AMD EPYC 9V45). Every other kernel
// those CPUs can run produces correct output, so when ParPar would pick it
// on its own, hand it the next-fastest kernel the CPU supports instead. A
// caller that forces the method explicitly is left alone.
static Galois16Methods resolve_auto_method() {
    Galois16Methods method = Galois16Mul::default_method();
    if (method != GF16_AFFINE_AVX512) return method;

    std::vector<Galois16Methods> avail = Galois16Mul::availableMethods(true);
    const Galois16Methods preferred[] = {
        GF16_SHUFFLE_VBMI, GF16_SHUFFLE_AVX512, GF16_SHUFFLE_AVX2, GF16_LOOKUP,
    };
    for (Galois16Methods m : preferred) {
        if (std::find(avail.begin(), avail.end(), m) != avail.end()) return m;
    }
    return method;
}

extern "C" {

parpar_gfproc_t* parpar_gfproc_new(size_t sliceSize, int numThreads,
                                    int method, unsigned inputGrouping,
                                    size_t chunkLen, unsigned stagingAreas) {
    ensure_gfmat_init();

    parpar_gfproc_t* proc = new(std::nothrow) parpar_gfproc_t;
    if (!proc) return nullptr;

    int sa = stagingAreas > 0 ? (int)stagingAreas : 2;
    proc->cpu = new(std::nothrow) PAR2ProcCPU(sa);
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
    Galois16Methods gfMethod = (method > 0) ? (Galois16Methods)method : resolve_auto_method();
    if (!proc->cpu->init(gfMethod, inputGrouping, chunkLen)) {
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

size_t parpar_gfproc_chunk_len(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return 0;
    return proc->cpu->getChunkLen();
}

unsigned parpar_gfproc_input_batch_size(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return 0;
    return proc->cpu->getInputBatchSize();
}

unsigned parpar_gfproc_alignment(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return 0;
    return proc->cpu->getAlignment();
}

unsigned parpar_gfproc_stride(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return 0;
    return proc->cpu->getStride();
}

size_t parpar_gfproc_alloc_slice_size(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return 0;
    return proc->cpu->getAllocSliceSize();
}

unsigned parpar_gfproc_staging_areas(parpar_gfproc_t* proc) {
    if (!proc || !proc->cpu) return 0;
    return proc->cpu->getStagingAreas();
}

} // extern "C"
