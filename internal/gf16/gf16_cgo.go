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
