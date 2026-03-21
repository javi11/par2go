// Package parpar provides Go bindings to ParPar's PAR2ProcCPU controller,
// which implements threaded PAR2-compatible Reed-Solomon encoding with
// optimized SIMD backends for 33+ CPU variants (SSE2, SSSE3, AVX, AVX2,
// AVX-512, GFNI, NEON, SVE, CLMul, RISC-V, etc.) and runtime CPU detection.
//
// Pre-built static libraries are committed per platform (see build-libs.yml).
// To rebuild from source: make -C internal/parpar libparpar_gf16.a
package parpar

/*
#cgo darwin LDFLAGS: ${SRCDIR}/libparpar_gf16_darwin.a -lstdc++ -lm
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/libparpar_gf16_linux_amd64.a -lstdc++ -lm -lpthread
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/libparpar_gf16_linux_arm64.a -lstdc++ -lm -lpthread
#cgo windows,amd64 LDFLAGS: ${SRCDIR}/libparpar_gf16_windows_amd64.a -lstdc++ -lm -lpthread
#include "bridge.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// GF16 method constants matching the Galois16Methods enum.
// Use GF16Auto (0) for runtime auto-detection of the fastest available method.
const (
	GF16Auto              = C.PARPAR_GF16_AUTO
	GF16Lookup            = C.PARPAR_GF16_LOOKUP
	GF16LookupSSE2        = C.PARPAR_GF16_LOOKUP_SSE2
	GF16Lookup3           = C.PARPAR_GF16_LOOKUP3
	GF16ShuffleNEON       = C.PARPAR_GF16_SHUFFLE_NEON
	GF16Shuffle128SVE     = C.PARPAR_GF16_SHUFFLE_128_SVE
	GF16Shuffle128SVE2    = C.PARPAR_GF16_SHUFFLE_128_SVE2
	GF16Shuffle2x128SVE2  = C.PARPAR_GF16_SHUFFLE2X_128_SVE2
	GF16Shuffle512SVE2    = C.PARPAR_GF16_SHUFFLE_512_SVE2
	GF16Shuffle128RVV     = C.PARPAR_GF16_SHUFFLE_128_RVV
	GF16ShuffleSSSE3      = C.PARPAR_GF16_SHUFFLE_SSSE3
	GF16ShuffleAVX        = C.PARPAR_GF16_SHUFFLE_AVX
	GF16ShuffleAVX2       = C.PARPAR_GF16_SHUFFLE_AVX2
	GF16ShuffleAVX512     = C.PARPAR_GF16_SHUFFLE_AVX512
	GF16ShuffleVBMI       = C.PARPAR_GF16_SHUFFLE_VBMI
	GF16Shuffle2xAVX2     = C.PARPAR_GF16_SHUFFLE2X_AVX2
	GF16Shuffle2xAVX512   = C.PARPAR_GF16_SHUFFLE2X_AVX512
	GF16XorSSE2           = C.PARPAR_GF16_XOR_SSE2
	GF16XorJitSSE2        = C.PARPAR_GF16_XOR_JIT_SSE2
	GF16XorJitAVX2        = C.PARPAR_GF16_XOR_JIT_AVX2
	GF16XorJitAVX512      = C.PARPAR_GF16_XOR_JIT_AVX512
	GF16AffineGFNI        = C.PARPAR_GF16_AFFINE_GFNI
	GF16AffineAVX2        = C.PARPAR_GF16_AFFINE_AVX2
	GF16AffineAVX10       = C.PARPAR_GF16_AFFINE_AVX10
	GF16AffineAVX512      = C.PARPAR_GF16_AFFINE_AVX512
	GF16Affine2xGFNI      = C.PARPAR_GF16_AFFINE2X_GFNI
	GF16Affine2xAVX2      = C.PARPAR_GF16_AFFINE2X_AVX2
	GF16Affine2xAVX10     = C.PARPAR_GF16_AFFINE2X_AVX10
	GF16Affine2xAVX512    = C.PARPAR_GF16_AFFINE2X_AVX512
	GF16ClmulNEON         = C.PARPAR_GF16_CLMUL_NEON
	GF16ClmulSHA3         = C.PARPAR_GF16_CLMUL_SHA3
	GF16ClmulSVE2         = C.PARPAR_GF16_CLMUL_SVE2
	GF16ClmulRVV          = C.PARPAR_GF16_CLMUL_RVV
)

// GfProc wraps ParPar's PAR2ProcCPU controller for PAR2-compatible
// Reed-Solomon encoding. It manages a pool of SIMD compute workers,
// a double-buffered input staging area, and the encoded output.
//
// Typical usage:
//
//	proc, _ := NewGfProc(sliceSize, 0)          // 0 = hardware_concurrency
//	proc.SetRecoverySlices([]uint16{0,1,2,...})  // before first Add
//	for i, data := range inputSlices {
//	    proc.Add(i, data)
//	}
//	proc.End()
//	for i := range recoverySlices {
//	    proc.GetOutput(i, recoveryBuf)
//	}
//	proc.Close()
type GfProc struct {
	handle *C.parpar_gfproc_t
}

// GfProcConfig configures a GfProc instance with full control over all
// PAR2ProcCPU parameters.
type GfProcConfig struct {
	// SliceSize is the block/slice size in bytes.
	SliceSize int
	// NumThreads sets the number of GF compute worker threads.
	// 0 selects hardware_concurrency() automatically.
	NumThreads int
	// Method selects the GF16 SIMD method. Use GF16Auto (0) for auto-detection.
	Method int
	// InputGrouping controls the input batch size (0 = auto, typically ~12).
	// Higher values use more memory but may improve throughput.
	InputGrouping int
	// ChunkLen controls the sub-slice chunk length for parallel processing
	// (0 = auto, uses the method's idealChunkSize).
	ChunkLen int
	// StagingAreas is the number of double-buffered staging areas (0 = default 2).
	// More areas can overlap I/O and compute better with many reader goroutines.
	StagingAreas int
}

// AddResult indicates the staging area state at the time of an Add call.
// Values match PAR2ProcBackendAddResult from the C++ controller.
type AddResult int

const (
	AddOK      AddResult = C.PARPAR_ADD_OK       // free staging slot, not busy
	AddOKBusy  AddResult = C.PARPAR_ADD_OK_BUSY  // can add, previous area still processing
	AddFull    AddResult = C.PARPAR_ADD_FULL      // had to wait for a staging slot
	AddAllFull AddResult = C.PARPAR_ADD_ALL_FULL  // controller-level: all full
)

// NewGfProc creates a new PAR2ProcCPU processor for slices of sliceSize bytes.
// numThreads sets the number of GF compute worker threads; use 0 to select
// hardware_concurrency() automatically.
//
// This is a convenience wrapper around NewGfProcWithConfig with default settings.
// Call SetRecoverySlices before the first Add.
func NewGfProc(sliceSize int, numThreads int) (*GfProc, error) {
	return NewGfProcWithConfig(GfProcConfig{
		SliceSize:  sliceSize,
		NumThreads: numThreads,
	})
}

// NewGfProcWithConfig creates a new PAR2ProcCPU processor with full control
// over all configuration parameters.
// Call SetRecoverySlices before the first Add.
func NewGfProcWithConfig(cfg GfProcConfig) (*GfProc, error) {
	h := C.parpar_gfproc_new(
		C.size_t(cfg.SliceSize),
		C.int(cfg.NumThreads),
		C.int(cfg.Method),
		C.uint(cfg.InputGrouping),
		C.size_t(cfg.ChunkLen),
		C.uint(cfg.StagingAreas),
	)
	if h == nil {
		return nil, ErrInitFailed
	}
	g := &GfProc{handle: h}
	runtime.SetFinalizer(g, (*GfProc).Close)
	return g, nil
}

// SetRecoverySlices configures the output recovery blocks.
// exponents[i] is the PAR2 exponent for the i-th recovery block;
// standard PAR2 uses {0, 1, 2, ..., numRecovery-1}.
// Must be called before the first Add.
func (g *GfProc) SetRecoverySlices(exponents []uint16) {
	if len(exponents) == 0 {
		C.parpar_gfproc_set_recovery_slices(g.handle, nil, 0)
		return
	}
	C.parpar_gfproc_set_recovery_slices(
		g.handle,
		(*C.uint16_t)(unsafe.Pointer(&exponents[0])),
		C.uint(len(exponents)),
	)
}

// Add submits input slice sliceNum (0-based) for encoding.
// It blocks until the prepare/transfer phase is complete (compute is async).
// Returns the staging area state at the time of the call.
func (g *GfProc) Add(sliceNum int, data []byte) AddResult {
	if len(data) == 0 {
		return AddOK
	}
	r := C.parpar_gfproc_add(
		g.handle,
		C.uint(sliceNum),
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
	)
	return AddResult(r)
}

// End signals that all inputs have been added and blocks until all compute
// worker threads have finished. Call before GetOutput.
func (g *GfProc) End() {
	C.parpar_gfproc_end(g.handle)
}

// GetOutput copies the recovery block at recoveryIdx into dst.
// dst must be at least sliceSize bytes. Blocks until the output is ready.
// Call after End.
func (g *GfProc) GetOutput(recoveryIdx int, dst []byte) {
	if len(dst) == 0 {
		return
	}
	C.parpar_gfproc_get_output(
		g.handle,
		C.uint(recoveryIdx),
		unsafe.Pointer(&dst[0]),
		C.size_t(len(dst)),
	)
}

// FreeMem releases the internal processing (output) buffer.
// Call after all outputs have been retrieved to reclaim memory.
func (g *GfProc) FreeMem() {
	C.parpar_gfproc_free_mem(g.handle)
}

// MethodName returns the name of the auto-detected SIMD method,
// e.g. "Shuffle (AVX2)" or "CLMul (NEON)".
func (g *GfProc) MethodName() string {
	return C.GoString(C.parpar_gfproc_method_name(g.handle))
}

// NumThreads returns the number of active GF compute worker threads.
func (g *GfProc) NumThreads() int {
	return int(C.parpar_gfproc_num_threads(g.handle))
}

// ChunkLen returns the actual sub-slice chunk length used for parallel processing.
func (g *GfProc) ChunkLen() int {
	return int(C.parpar_gfproc_chunk_len(g.handle))
}

// InputBatchSize returns the actual input batch size.
func (g *GfProc) InputBatchSize() int {
	return int(C.parpar_gfproc_input_batch_size(g.handle))
}

// Alignment returns the buffer alignment requirement in bytes.
func (g *GfProc) Alignment() int {
	return int(C.parpar_gfproc_alignment(g.handle))
}

// Stride returns the SIMD stride multiplier in bytes.
func (g *GfProc) Stride() int {
	return int(C.parpar_gfproc_stride(g.handle))
}

// AllocSliceSize returns the padded slice size including alignment.
func (g *GfProc) AllocSliceSize() int {
	return int(C.parpar_gfproc_alloc_slice_size(g.handle))
}

// StagingAreas returns the number of double-buffered staging areas.
func (g *GfProc) StagingAreas() int {
	return int(C.parpar_gfproc_staging_areas(g.handle))
}

// Close frees all C++ resources. After Close the GfProc must not be used.
func (g *GfProc) Close() {
	if g.handle != nil {
		C.parpar_gfproc_free(g.handle)
		g.handle = nil
	}
	runtime.SetFinalizer(g, nil)
}

// ErrInitFailed is returned when PAR2ProcCPU initialization fails.
type ErrType string

func (e ErrType) Error() string { return string(e) }

const ErrInitFailed = ErrType("parpar: failed to initialize PAR2ProcCPU")
