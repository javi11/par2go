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

// Create a new PAR2ProcCPU processor for the given slice size.
// numThreads <= 0 selects hardware_concurrency().
// Returns NULL on allocation or initialization failure.
parpar_gfproc_t* parpar_gfproc_new(size_t sliceSize, int numThreads);

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

#ifdef __cplusplus
}
#endif

#endif // PARPAR_BRIDGE_H
