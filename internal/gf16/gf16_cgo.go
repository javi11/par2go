package gf16

import (
	"sync"

	"github.com/javi11/par2go/internal/parpar"
)

// MulAccumulate routes through ParPar's optimized SIMD backends which handle
// CPU detection and method dispatch across 33+ SIMD variants.

var (
	gf16Once    sync.Once
	gf16Inst    *parpar.GF16
	scratchPool sync.Pool
)

func initParPar() {
	var err error
	gf16Inst, err = parpar.New()
	if err != nil {
		panic("gf16: failed to initialize ParPar: " + err.Error())
	}
	scratchPool.New = func() any {
		return gf16Inst.NewScratch()
	}
}

// mulAccumulate delegates to ParPar's SIMD-optimized GF(2^16) multiply-accumulate.
func mulAccumulate(dst, src []byte, factor uint16) {
	gf16Once.Do(initParPar)
	s := scratchPool.Get().(*parpar.Scratch)
	gf16Inst.MulAdd(dst, src, factor, s)
	scratchPool.Put(s)
}

// AlignedSlice allocates a byte slice aligned to the SIMD requirements of the
// underlying GF16 instance. The returned slice must be freed with FreeAligned.
func AlignedSlice(size int) []byte {
	gf16Once.Do(initParPar)
	return gf16Inst.AlignedSlice(size)
}

// FreeAligned frees a slice previously allocated with AlignedSlice.
func FreeAligned(b []byte) {
	parpar.FreeAligned(b)
}

// BatchAccumulator holds per-goroutine state for the prepare-once/accumulate-many
// pattern. It pre-allocates an aligned input-preparation buffer so that each
// input slice is prepared exactly once, and MulAddPacked is used for all
// accumulations with no per-call heap allocation.
//
// Create one per goroutine; it is NOT safe for concurrent use.
type BatchAccumulator struct {
	prepBuf []byte
	scratch *parpar.Scratch
}

// NewBatchAccumulator creates a BatchAccumulator with a pre-allocated aligned
// buffer of the given size. The caller must call Free when done.
func NewBatchAccumulator(size int) *BatchAccumulator {
	gf16Once.Do(initParPar)
	s := scratchPool.Get().(*parpar.Scratch)
	return &BatchAccumulator{
		prepBuf: gf16Inst.AlignedSlice(size),
		scratch: s,
	}
}

// Free releases the aligned buffer and returns the scratch to the pool.
func (ba *BatchAccumulator) Free() {
	parpar.FreeAligned(ba.prepBuf)
	ba.prepBuf = nil
	scratchPool.Put(ba.scratch)
	ba.scratch = nil
}

// PrepareInput converts src into the aligned packed format stored in the
// accumulator's internal buffer. Call this once per input slice before
// calling AccumulatePrepared for each recovery block.
// len(src) must equal the size passed to NewBatchAccumulator.
func (ba *BatchAccumulator) PrepareInput(src []byte) {
	if gf16Inst.NeedsPrepare() {
		gf16Inst.Prepare(ba.prepBuf, src)
	} else {
		copy(ba.prepBuf, src)
	}
}

// AccumulatePrepared adds the prepared input into dst multiplied by factor.
// dst must be aligned (allocated via AlignedSlice).
// len(dst) must equal the size passed to NewBatchAccumulator.
// This call performs no heap allocation.
func (ba *BatchAccumulator) AccumulatePrepared(dst []byte, factor uint16) {
	if factor == 0 {
		return
	}
	if factor == 1 {
		xorBytes(dst, ba.prepBuf[:len(dst)])
		return
	}
	if gf16Inst.NeedsPrepare() {
		gf16Inst.MulAddPacked(dst, ba.prepBuf[:len(dst)], factor, ba.scratch)
	} else {
		gf16Inst.MulAdd(dst, ba.prepBuf[:len(dst)], factor, ba.scratch)
	}
}

// FinishBlock converts a recovery block from packed format back to raw format
// in-place. Call once per recovery block after all inputs have been accumulated.
// No-op when NeedsPrepare() is false.
func (ba *BatchAccumulator) FinishBlock(buf []byte) {
	if gf16Inst.NeedsPrepare() {
		gf16Inst.Finish(buf)
	}
}
