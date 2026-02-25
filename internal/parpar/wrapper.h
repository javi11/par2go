// wrapper.h - Thin C API around ParPar's Galois16Mul C++ class.
// This enables cgo to call into ParPar without exposing C++ types.

#ifndef PARPAR_WRAPPER_H
#define PARPAR_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a Galois16Mul instance.
typedef struct parpar_gf16 parpar_gf16_t;

// Create a new GF16 multiplier. method=0 means auto-detect best.
parpar_gf16_t* parpar_gf16_new(int method);

// Free a GF16 multiplier.
void parpar_gf16_free(parpar_gf16_t* gf);

// Return the name of the selected method (e.g. "Shuffle (AVX2)").
const char* parpar_gf16_method_name(parpar_gf16_t* gf);

// Return the required memory alignment for buffers.
size_t parpar_gf16_alignment(parpar_gf16_t* gf);

// Return the stride (minimum processing granularity).
size_t parpar_gf16_stride(parpar_gf16_t* gf);

// Allocate thread-local mutable scratch memory.
// Each goroutine/thread needs its own scratch.
void* parpar_gf16_scratch_alloc(parpar_gf16_t* gf);

// Free thread-local scratch memory.
void parpar_gf16_scratch_free(parpar_gf16_t* gf, void* scratch);

// Core operation: dst[i] ^= src[i] * coefficient in GF(2^16).
// len must be a multiple of stride. src and dst must be aligned to alignment.
// scratch must be from parpar_gf16_scratch_alloc (or NULL if not needed).
void parpar_gf16_muladd(parpar_gf16_t* gf, void* dst, const void* src,
                         size_t len, uint16_t coefficient, void* scratch);

// Allocate memory with the specified alignment.
void* parpar_aligned_alloc(size_t alignment, size_t size);

// Free memory from parpar_aligned_alloc.
void parpar_aligned_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // PARPAR_WRAPPER_H
