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

    gf->mul = new(std::nothrow) Galois16Mul(static_cast<Galois16Methods>(method));
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

void parpar_gf16_muladd(parpar_gf16_t* gf, void* dst, const void* src,
                         size_t len, uint16_t coefficient, void* scratch) {
    if (!gf || !gf->mul || len == 0) return;
    if (coefficient == 0) return;

    // Use the _mul_add function pointer through the public mul_add method.
    // Note: mul_add requires PARPAR_INVERT_SUPPORT to be defined.
    gf->mul->mul_add(dst, src, len, coefficient, scratch);
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
