// bridge.h - C API around ParPar's PAR2ProcCPU controller.
// This enables cgo to call into ParPar's high-level PAR2 encoding pipeline
// without exposing C++ types.

#ifndef PARPAR_BRIDGE_H
#define PARPAR_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a PAR2ProcCPU instance.
typedef struct parpar_gfproc parpar_gfproc_t;

// Add result codes (match PAR2ProcBackendAddResult enum values).
#define PARPAR_ADD_OK       0
#define PARPAR_ADD_OK_BUSY  1
#define PARPAR_ADD_FULL     2
#define PARPAR_ADD_ALL_FULL 3

// GF16 method constants (match Galois16Methods enum).
#define PARPAR_GF16_AUTO                 0
#define PARPAR_GF16_LOOKUP               1
#define PARPAR_GF16_LOOKUP_SSE2          2
#define PARPAR_GF16_LOOKUP3              3
#define PARPAR_GF16_SHUFFLE_NEON         4
#define PARPAR_GF16_SHUFFLE_128_SVE      5
#define PARPAR_GF16_SHUFFLE_128_SVE2     6
#define PARPAR_GF16_SHUFFLE2X_128_SVE2   7
#define PARPAR_GF16_SHUFFLE_512_SVE2     8
#define PARPAR_GF16_SHUFFLE_128_RVV      9
#define PARPAR_GF16_SHUFFLE_SSSE3        10
#define PARPAR_GF16_SHUFFLE_AVX          11
#define PARPAR_GF16_SHUFFLE_AVX2         12
#define PARPAR_GF16_SHUFFLE_AVX512       13
#define PARPAR_GF16_SHUFFLE_VBMI         14
#define PARPAR_GF16_SHUFFLE2X_AVX2       15
#define PARPAR_GF16_SHUFFLE2X_AVX512     16
#define PARPAR_GF16_XOR_SSE2             17
#define PARPAR_GF16_XOR_JIT_SSE2         18
#define PARPAR_GF16_XOR_JIT_AVX2         19
#define PARPAR_GF16_XOR_JIT_AVX512       20
#define PARPAR_GF16_AFFINE_GFNI          21
#define PARPAR_GF16_AFFINE_AVX2          22
#define PARPAR_GF16_AFFINE_AVX10         23
#define PARPAR_GF16_AFFINE_AVX512        24
#define PARPAR_GF16_AFFINE2X_GFNI        25
#define PARPAR_GF16_AFFINE2X_AVX2        26
#define PARPAR_GF16_AFFINE2X_AVX10       27
#define PARPAR_GF16_AFFINE2X_AVX512      28
#define PARPAR_GF16_CLMUL_NEON           29
#define PARPAR_GF16_CLMUL_SHA3           30
#define PARPAR_GF16_CLMUL_SVE2           31
#define PARPAR_GF16_CLMUL_RVV            32

// Create a new PAR2ProcCPU processor for the given slice size.
//   numThreads   <= 0: use hardware_concurrency()
//   method:       PARPAR_GF16_* constant (0 = auto-detect)
//   inputGrouping: input batch size (0 = auto, typically ~12)
//   chunkLen:      sub-slice chunk length (0 = auto, method's idealChunkSize)
//   stagingAreas:  number of double-buffered staging areas (0 = default 2)
// Returns NULL on allocation or initialization failure.
parpar_gfproc_t* parpar_gfproc_new(size_t sliceSize, int numThreads,
                                    int method, unsigned inputGrouping,
                                    size_t chunkLen, unsigned stagingAreas);

// Free a processor and release all C++ resources.
void parpar_gfproc_free(parpar_gfproc_t* proc);

// Set recovery slice exponents (must be called before the first Add).
// exponents[i] is the PAR2 exponent for output recovery block i.
// Typical use: exponents = {0, 1, 2, ..., numRecovery-1}.
void parpar_gfproc_set_recovery_slices(parpar_gfproc_t* proc, const uint16_t* exponents, unsigned count);

// Change the number of compute worker threads.
void parpar_gfproc_set_num_threads(parpar_gfproc_t* proc, int n);

// Add one input slice. sliceNum is the 0-based input block index (uint16).
// Blocks until the prepare/transfer phase for this input is complete.
// Returns one of the PARPAR_ADD_* codes indicating the staging area state
// at the time of the call (informational; full waits are handled internally).
int parpar_gfproc_add(parpar_gfproc_t* proc, unsigned sliceNum, const void* data, size_t len);

// Signal end of input. Blocks until all pending compute work is complete.
void parpar_gfproc_end(parpar_gfproc_t* proc);

// Copy recovery slice at recoveryIdx into dst (must be >= sliceSize bytes).
// Must be called after parpar_gfproc_end. Blocks until the output is ready.
void parpar_gfproc_get_output(parpar_gfproc_t* proc, unsigned recoveryIdx, void* dst, size_t len);

// Free internal processing memory. Call after all outputs have been retrieved.
// The processor may be reconfigured with setRecoverySlices and reused.
void parpar_gfproc_free_mem(parpar_gfproc_t* proc);

// Return the name of the selected SIMD method (e.g. "Shuffle (AVX2)").
const char* parpar_gfproc_method_name(parpar_gfproc_t* proc);

// Return the number of active compute threads.
unsigned parpar_gfproc_num_threads(parpar_gfproc_t* proc);

// Query the actual chunk length after auto-calculation.
size_t parpar_gfproc_chunk_len(parpar_gfproc_t* proc);

// Query the actual input batch size.
unsigned parpar_gfproc_input_batch_size(parpar_gfproc_t* proc);

// Query the buffer alignment requirement.
unsigned parpar_gfproc_alignment(parpar_gfproc_t* proc);

// Query the SIMD stride multiplier.
unsigned parpar_gfproc_stride(parpar_gfproc_t* proc);

// Query the padded slice size including alignment.
size_t parpar_gfproc_alloc_slice_size(parpar_gfproc_t* proc);

// Query the number of staging areas.
unsigned parpar_gfproc_staging_areas(parpar_gfproc_t* proc);

#ifdef __cplusplus
}
#endif

#endif // PARPAR_BRIDGE_H
