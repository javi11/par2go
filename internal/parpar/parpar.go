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
#cgo windows,amd64 LDFLAGS: ${SRCDIR}/libparpar_gf16_windows_amd64.a -lstdc++ -lm
#include "bridge.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
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
// Call SetRecoverySlices before the first Add.
func NewGfProc(sliceSize int, numThreads int) (*GfProc, error) {
	h := C.parpar_gfproc_new(C.size_t(sliceSize), C.int(numThreads))
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
