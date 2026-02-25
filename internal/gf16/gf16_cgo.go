//go:build cgo && !purego

package gf16

import (
	"sync"

	"github.com/javi11/par2go/internal/parpar"
)

// When cgo is available, we use ParPar's optimized SIMD backends for
// MulAccumulate instead of our hand-written assembly. ParPar handles its
// own CPU detection and method dispatch across 33+ SIMD variants.

var useSIMD = false // ParPar dispatches internally; we bypass the old SIMD path

// mulAccumulateSIMD is unused in the cgo path but required by gf16.go's dispatch.
func mulAccumulateSIMD(_, _ []byte, _ *MulAccTables) {}

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

// mulAccumulate routes through ParPar's SIMD-optimized GF(2^16) multiply-accumulate.
func mulAccumulate(dst, src []byte, factor uint16) {
	gf16Once.Do(initParPar)
	s := scratchPool.Get().(*parpar.Scratch)
	gf16Inst.MulAdd(dst, src, factor, s)
	scratchPool.Put(s)
}
