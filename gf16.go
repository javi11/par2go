package par2go

import "github.com/javi11/par2go/internal/parpar"

// GF16 kernel identifiers for Options.Method. GF16Auto lets the bridge pick
// the fastest kernel the CPU supports (excluding kernels known to crash);
// any other value forces that kernel and is used even if it is known-broken.
const (
	GF16Auto             = parpar.GF16Auto
	GF16Lookup           = parpar.GF16Lookup
	GF16LookupSSE2       = parpar.GF16LookupSSE2
	GF16Lookup3          = parpar.GF16Lookup3
	GF16ShuffleNEON      = parpar.GF16ShuffleNEON
	GF16Shuffle128SVE    = parpar.GF16Shuffle128SVE
	GF16Shuffle128SVE2   = parpar.GF16Shuffle128SVE2
	GF16Shuffle2x128SVE2 = parpar.GF16Shuffle2x128SVE2
	GF16Shuffle512SVE2   = parpar.GF16Shuffle512SVE2
	GF16Shuffle128RVV    = parpar.GF16Shuffle128RVV
	GF16ShuffleSSSE3     = parpar.GF16ShuffleSSSE3
	GF16ShuffleAVX       = parpar.GF16ShuffleAVX
	GF16ShuffleAVX2      = parpar.GF16ShuffleAVX2
	GF16ShuffleAVX512    = parpar.GF16ShuffleAVX512
	GF16ShuffleVBMI      = parpar.GF16ShuffleVBMI
	GF16Shuffle2xAVX2    = parpar.GF16Shuffle2xAVX2
	GF16Shuffle2xAVX512  = parpar.GF16Shuffle2xAVX512
	GF16XorSSE2          = parpar.GF16XorSSE2
	GF16XorJitSSE2       = parpar.GF16XorJitSSE2
	GF16XorJitAVX2       = parpar.GF16XorJitAVX2
	GF16XorJitAVX512     = parpar.GF16XorJitAVX512
	GF16AffineGFNI       = parpar.GF16AffineGFNI
	GF16AffineAVX2       = parpar.GF16AffineAVX2
	GF16AffineAVX10      = parpar.GF16AffineAVX10
	GF16AffineAVX512     = parpar.GF16AffineAVX512
	GF16Affine2xGFNI     = parpar.GF16Affine2xGFNI
	GF16Affine2xAVX2     = parpar.GF16Affine2xAVX2
	GF16Affine2xAVX10    = parpar.GF16Affine2xAVX10
	GF16Affine2xAVX512   = parpar.GF16Affine2xAVX512
	GF16ClmulNEON        = parpar.GF16ClmulNEON
	GF16ClmulSHA3        = parpar.GF16ClmulSHA3
	GF16ClmulSVE2        = parpar.GF16ClmulSVE2
	GF16ClmulRVV         = parpar.GF16ClmulRVV
)
