// Package parpar provides Go bindings to ParPar's GF(2^16) multiply-accumulate
// implementation, which includes optimized SIMD backends for 33+ CPU variants
// (SSE2, SSSE3, AVX, AVX2, AVX-512, GFNI, NEON, SVE, CLMul, RISC-V, etc.)
// with runtime CPU detection and automatic dispatch.
//
// Pre-built static libraries are committed per platform (see build-libs.yml).
// To rebuild from source: make -C internal/parpar libparpar_gf16.a
package parpar

/*
#cgo darwin LDFLAGS: ${SRCDIR}/libparpar_gf16_darwin.a -lstdc++ -lm
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/libparpar_gf16_linux_amd64.a -lstdc++ -lm
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/libparpar_gf16_linux_arm64.a -lstdc++ -lm
#cgo windows,amd64 LDFLAGS: ${SRCDIR}/libparpar_gf16_windows_amd64.a -lstdc++ -lm
#include "wrapper.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"

import (
	"runtime"
	"sync"
	"unsafe"
)

// GF16 wraps a ParPar Galois16Mul instance with automatic SIMD dispatch.
// Create one per application; it is safe for concurrent use.
type GF16 struct {
	handle *C.parpar_gf16_t
}

// New creates a new GF16 multiplier with automatic method detection.
func New() (*GF16, error) {
	h := C.parpar_gf16_new(0) // 0 = GF16_AUTO
	if h == nil {
		return nil, ErrInitFailed
	}
	gf := &GF16{handle: h}
	runtime.SetFinalizer(gf, (*GF16).close)
	return gf, nil
}

func (gf *GF16) close() {
	if gf.handle != nil {
		C.parpar_gf16_free(gf.handle)
		gf.handle = nil
	}
}

// Close frees the underlying C++ resources. After Close, the GF16 must not be used.
func (gf *GF16) Close() {
	gf.close()
	runtime.SetFinalizer(gf, nil)
}

// MethodName returns the name of the auto-detected SIMD method, e.g. "Shuffle (AVX2)".
func (gf *GF16) MethodName() string {
	return C.GoString(C.parpar_gf16_method_name(gf.handle))
}

// Alignment returns the required byte alignment for src/dst buffers.
func (gf *GF16) Alignment() int {
	return int(C.parpar_gf16_alignment(gf.handle))
}

// Stride returns the minimum processing granularity. Input length should be
// a multiple of stride for optimal performance.
func (gf *GF16) Stride() int {
	return int(C.parpar_gf16_stride(gf.handle))
}

// Scratch holds thread-local mutable scratch memory required by some methods
// (e.g. XOR JIT). Each goroutine calling MulAdd should use its own Scratch.
// For methods that don't need scratch, the internal pointer is nil and this
// is a no-op wrapper.
type Scratch struct {
	gf  *GF16
	ptr unsafe.Pointer
}

// NewScratch allocates scratch memory for use with MulAdd.
// The caller must call Free when done.
func (gf *GF16) NewScratch() *Scratch {
	return &Scratch{
		gf:  gf,
		ptr: unsafe.Pointer(C.parpar_gf16_scratch_alloc(gf.handle)),
	}
}

// Free releases the scratch memory.
func (s *Scratch) Free() {
	if s.ptr != nil {
		C.parpar_gf16_scratch_free(s.gf.handle, s.ptr)
		s.ptr = nil
	}
}

// MulAdd computes dst[i] ^= src[i] * coefficient in GF(2^16) for all i.
// dst and src are treated as slices of little-endian uint16 values.
//
// For best performance:
//   - len(dst) == len(src) and both should be multiples of Stride()
//   - use a per-goroutine Scratch (from a sync.Pool)
//
// The C wrapper handles alignment and data format conversion (prepare/finish)
// transparently. Non-stride-multiple remainders are handled with a Go scalar
// fallback.
func (gf *GF16) MulAdd(dst, src []byte, coefficient uint16, scratch *Scratch) {
	if len(src) == 0 || coefficient == 0 {
		return
	}

	stride := gf.Stride()
	n := len(src)
	aligned := n - (n % stride)

	var scratchPtr unsafe.Pointer
	if scratch != nil {
		scratchPtr = scratch.ptr
	}

	if aligned > 0 {
		// The C wrapper handles alignment and prepare/finish internally.
		C.parpar_gf16_muladd(
			gf.handle,
			unsafe.Pointer(&dst[0]),
			unsafe.Pointer(&src[0]),
			C.size_t(aligned), C.uint16_t(coefficient), scratchPtr,
		)
	}

	// Handle remainder with scalar Go code
	if aligned < n {
		mulAddScalarTail(dst[aligned:], src[aligned:], coefficient)
	}
}

// NeedsPrepare returns true if the selected SIMD method requires data to be
// converted to an internal packed format before processing. When true, use
// Prepare/MulAddPacked/Finish for batch operations to avoid repeated
// alloc/copy overhead in the hot path.
func (gf *GF16) NeedsPrepare() bool {
	return C.parpar_gf16_needs_prepare(gf.handle) != 0
}

// Prepare converts src data to the packed format required by MulAddPacked.
// dst must be aligned to Alignment() bytes. dst and src may be the same pointer.
// len(dst) must equal len(src).
func (gf *GF16) Prepare(dst, src []byte) {
	if len(src) == 0 {
		return
	}
	C.parpar_gf16_prepare(
		gf.handle,
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&src[0]),
		C.size_t(len(src)),
	)
}

// Finish converts packed data back to raw format in-place.
// buf must be aligned to Alignment() bytes.
func (gf *GF16) Finish(buf []byte) {
	if len(buf) == 0 {
		return
	}
	C.parpar_gf16_finish(
		gf.handle,
		unsafe.Pointer(&buf[0]),
		C.size_t(len(buf)),
	)
}

// MulAddPacked computes dst[i] ^= src[i] * coefficient in GF(2^16) on
// already-prepared (packed) data. Both dst and src must be aligned to
// Alignment() bytes. Use Prepare before calling and Finish after all
// accumulation is complete for a given buffer.
func (gf *GF16) MulAddPacked(dst, src []byte, coefficient uint16, scratch *Scratch) {
	if len(src) == 0 || coefficient == 0 {
		return
	}
	var scratchPtr unsafe.Pointer
	if scratch != nil {
		scratchPtr = scratch.ptr
	}
	C.parpar_gf16_muladd_packed(
		gf.handle,
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&src[0]),
		C.size_t(len(src)),
		C.uint16_t(coefficient),
		scratchPtr,
	)
}

// mulAddScalarTail handles the tail bytes that don't fill a full stride.
// It uses log/exp table lookups, same algorithm as the pure-Go fallback.
func mulAddScalarTail(dst, src []byte, factor uint16) {
	logFactor := logTable[factor]
	for i := 0; i+1 < len(src); i += 2 {
		val := uint16(src[i]) | uint16(src[i+1])<<8
		if val == 0 {
			continue
		}
		product := expTable[uint32(logTable[val])+uint32(logFactor)]
		dst[i] ^= byte(product)
		dst[i+1] ^= byte(product >> 8)
	}
}

// GF(2^16) log/exp tables for scalar tail processing.
// These are identical to the tables in the gf16 package.
const (
	gf16Polynomial = 0x1100B
)

var (
	logTable [65536]uint16
	expTable [2 * 65535]uint16
	initOnce sync.Once
)

func init() {
	initOnce.Do(buildTables)
}

func buildTables() {
	var val uint32 = 1
	for i := 0; i < 65535; i++ {
		expTable[i] = uint16(val)
		expTable[i+65535] = uint16(val)
		logTable[val] = uint16(i)
		val <<= 1
		if val&0x10000 != 0 {
			val ^= gf16Polynomial
		}
	}
}

// AlignedSlice allocates a byte slice of the given size with memory aligned
// to the SIMD requirements of this GF16 instance. Using aligned slices with
// MulAdd avoids the overhead of copying through aligned temporaries.
// The returned slice must be freed with FreeAligned when no longer needed.
func (gf *GF16) AlignedSlice(size int) []byte {
	alignment := gf.Alignment()
	ptr := C.parpar_aligned_alloc(C.size_t(alignment), C.size_t(size))
	if ptr == nil {
		return nil
	}
	C.memset(ptr, 0, C.size_t(size))
	return unsafe.Slice((*byte)(ptr), size)
}

// FreeAligned frees a slice previously allocated with AlignedSlice.
func FreeAligned(b []byte) {
	if len(b) > 0 {
		C.parpar_aligned_free(unsafe.Pointer(&b[0]))
	}
}

// ErrInitFailed is returned when ParPar GF16 initialization fails.
type ErrType string

func (e ErrType) Error() string { return string(e) }

const ErrInitFailed = ErrType("parpar: failed to initialize GF16 multiplier")
