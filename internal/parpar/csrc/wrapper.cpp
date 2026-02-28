// wrapper.cpp - Thin C wrapper around ParPar's Galois16Mul C++ class.

#include "wrapper.h"
#include "vendor/gf16/gf16mul.h"
#include "vendor/src/platform.h"

#include <cstdlib>
#include <cstring>

struct parpar_gf16 {
    Galois16Mul* mul;
};

extern "C" {

parpar_gf16_t* parpar_gf16_new(int method) {
    parpar_gf16_t* gf = new(std::nothrow) parpar_gf16_t;
    if (!gf) return NULL;

    Galois16Methods m = static_cast<Galois16Methods>(method);
    if (m == GF16_AUTO) {
        // Use forInvert=true to avoid XOR-JIT methods which require
        // aligned buffers and writable-executable memory. Since Go
        // allocates buffers with its own allocator (not aligned to SIMD
        // requirements), non-JIT methods (Shuffle, Affine, CLMul) are
        // more compatible and still fast.
        m = Galois16Mul::default_method(1048576, 32768, 65535, true);
    }

    gf->mul = new(std::nothrow) Galois16Mul(m);
    if (!gf->mul) {
        delete gf;
        return NULL;
    }
    return gf;
}

void parpar_gf16_free(parpar_gf16_t* gf) {
    if (!gf) return;
    delete gf->mul;
    delete gf;
}

const char* parpar_gf16_method_name(parpar_gf16_t* gf) {
    if (!gf || !gf->mul) return "unknown";
    return gf->mul->info().name;
}

size_t parpar_gf16_alignment(parpar_gf16_t* gf) {
    if (!gf || !gf->mul) return 2;
    return gf->mul->info().alignment;
}

size_t parpar_gf16_stride(parpar_gf16_t* gf) {
    if (!gf || !gf->mul) return 2;
    return gf->mul->info().stride;
}

void* parpar_gf16_scratch_alloc(parpar_gf16_t* gf) {
    if (!gf || !gf->mul) return NULL;
    return gf->mul->mutScratch_alloc();
}

void parpar_gf16_scratch_free(parpar_gf16_t* gf, void* scratch) {
    if (!gf || !gf->mul) return;
    gf->mul->mutScratch_free(scratch);
}

int parpar_gf16_needs_prepare(parpar_gf16_t* gf) {
    if (!gf || !gf->mul) return 0;
    return gf->mul->needPrepare() ? 1 : 0;
}

void parpar_gf16_muladd(parpar_gf16_t* gf, void* dst, const void* src,
                         size_t len, uint16_t coefficient, void* scratch) {
    if (!gf || !gf->mul || len == 0) return;
    if (coefficient == 0) return;

    if (!gf->mul->needPrepare()) {
        // Methods that don't need prepare (e.g. Lookup) work on raw data.
        gf->mul->mul_add(dst, src, len, coefficient, scratch);
        return;
    }

    // Methods like Shuffle and Affine operate on "prepared" (packed) data
    // where high and low bytes of uint16 values are separated. We need to
    // transform the data before mul_add and transform it back after.
    size_t alignment = gf->mul->info().alignment;

    void* prepSrc = NULL;
    void* prepDst = NULL;
    ALIGN_ALLOC(prepSrc, len, alignment);
    ALIGN_ALLOC(prepDst, len, alignment);

    if (!prepSrc || !prepDst) {
        if (prepSrc) ALIGN_FREE(prepSrc);
        if (prepDst) ALIGN_FREE(prepDst);
        return;
    }

    // Convert to packed format (also copies to aligned buffers)
    gf->mul->prepare(prepSrc, src, len);
    gf->mul->prepare(prepDst, dst, len);

    // Multiply-accumulate in packed format
    gf->mul->mul_add(prepDst, prepSrc, len, coefficient, scratch);

    // Convert back to natural format (in-place)
    gf->mul->finish(prepDst, len);

    // Copy result back
    memcpy(dst, prepDst, len);

    ALIGN_FREE(prepSrc);
    ALIGN_FREE(prepDst);
}

void parpar_gf16_muladd_packed(parpar_gf16_t* gf, void* dst, const void* src,
                                size_t len, uint16_t coefficient, void* scratch) {
    if (!gf || !gf->mul || len == 0) return;
    if (coefficient == 0) return;
    gf->mul->mul_add(dst, src, len, coefficient, scratch);
}

void parpar_gf16_muladd_packed_multi(parpar_gf16_t* gf, void* dsts_array,
                                     size_t dst_count, const void* src,
                                     const uint16_t* coefficients, size_t len,
                                     void* scratch) {
    if (!gf || !gf->mul || len == 0 || dst_count == 0) return;
    void** dsts = static_cast<void**>(dsts_array);
    for (size_t i = 0; i < dst_count; i++) {
        if (coefficients[i] != 0) {
            gf->mul->mul_add(dsts[i], src, len, coefficients[i], scratch);
        }
    }
}

void parpar_gf16_prepare(parpar_gf16_t* gf, void* dst, const void* src, size_t len) {
    if (!gf || !gf->mul || len == 0) return;
    gf->mul->prepare(dst, src, len);
}

void parpar_gf16_finish(parpar_gf16_t* gf, void* dst, size_t len) {
    if (!gf || !gf->mul || len == 0) return;
    gf->mul->finish(dst, len);
}

void* parpar_aligned_alloc(size_t alignment, size_t size) {
    void* ptr = NULL;
    ALIGN_ALLOC(ptr, size, alignment);
    return ptr;
}

void parpar_aligned_free(void* ptr) {
    ALIGN_FREE(ptr);
}

} // extern "C"
