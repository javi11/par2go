package parpar

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"testing"
)

func TestNew(t *testing.T) {
	gf, err := New()
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer gf.Close()

	name := gf.MethodName()
	if name == "" || name == "unknown" {
		t.Errorf("unexpected method name: %q", name)
	}
	t.Logf("Method: %s, Alignment: %d, Stride: %d", name, gf.Alignment(), gf.Stride())
}

func TestMulAddZeroCoefficient(t *testing.T) {
	gf, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()

	dst := make([]byte, 64)
	src := make([]byte, 64)
	for i := range src {
		src[i] = byte(i)
	}
	orig := make([]byte, 64)
	copy(orig, dst)

	scratch := gf.NewScratch()
	defer scratch.Free()

	gf.MulAdd(dst, src, 0, scratch)
	if !bytes.Equal(dst, orig) {
		t.Error("MulAdd with coefficient=0 should be a no-op")
	}
}

func TestMulAddCrossValidation(t *testing.T) {
	gf, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()

	scratch := gf.NewScratch()
	defer scratch.Free()

	// Use sizes that are multiples of common strides
	stride := gf.Stride()
	sizes := []int{stride, stride * 2, stride * 4, stride * 16, stride * 64}

	factors := []uint16{1, 2, 3, 0x1234, 0xFFFF, 0x8000}

	for _, size := range sizes {
		for _, factor := range factors {
			t.Run(fmt.Sprintf("size=%d_factor=0x%04X", size, factor), func(t *testing.T) {
				src := make([]byte, size)
				for i := range src {
					src[i] = byte(rand.IntN(256))
				}

				// ParPar result
				dstParpar := make([]byte, size)
				gf.MulAdd(dstParpar, src, factor, scratch)

				// Reference scalar result
				dstRef := make([]byte, size)
				refMulAccumulate(dstRef, src, factor)

				if !bytes.Equal(dstParpar, dstRef) {
					// Find first mismatch
					for i := range dstParpar {
						if dstParpar[i] != dstRef[i] {
							t.Errorf("mismatch at byte %d: parpar=0x%02X ref=0x%02X", i, dstParpar[i], dstRef[i])
							break
						}
					}
				}
			})
		}
	}
}

func TestMulAddNonStrideSize(t *testing.T) {
	gf, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()

	scratch := gf.NewScratch()
	defer scratch.Free()

	// Test sizes that are NOT multiples of stride but are multiples of 2
	stride := gf.Stride()
	sizes := []int{2, 6, 14, 18, 30, stride + 2, stride*2 + 6}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			src := make([]byte, size)
			for i := range src {
				src[i] = byte(rand.IntN(256))
			}

			dstParpar := make([]byte, size)
			gf.MulAdd(dstParpar, src, 0x5678, scratch)

			dstRef := make([]byte, size)
			refMulAccumulate(dstRef, src, 0x5678)

			if !bytes.Equal(dstParpar, dstRef) {
				for i := range dstParpar {
					if dstParpar[i] != dstRef[i] {
						t.Errorf("mismatch at byte %d: parpar=0x%02X ref=0x%02X", i, dstParpar[i], dstRef[i])
						break
					}
				}
			}
		})
	}
}

func TestMulAddXORSemantics(t *testing.T) {
	gf, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()

	scratch := gf.NewScratch()
	defer scratch.Free()

	stride := gf.Stride()
	size := stride * 4

	src := make([]byte, size)
	for i := range src {
		src[i] = byte(rand.IntN(256))
	}

	// Pre-fill dst with non-zero data
	dst := make([]byte, size)
	for i := range dst {
		dst[i] = byte(rand.IntN(256))
	}
	origDst := make([]byte, size)
	copy(origDst, dst)

	gf.MulAdd(dst, src, 0x42, scratch)

	// Verify XOR semantics: dst should be origDst XOR (src * factor)
	expected := make([]byte, size)
	copy(expected, origDst)
	refMulAccumulate(expected, src, 0x42)

	if !bytes.Equal(dst, expected) {
		t.Error("MulAdd should XOR into dst, not overwrite")
	}
}

func TestMulAddLargeBuffer(t *testing.T) {
	gf, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()

	scratch := gf.NewScratch()
	defer scratch.Free()

	// 768KB - typical PAR2 slice size
	size := 768 * 1024
	src := make([]byte, size)
	for i := range src {
		src[i] = byte(rand.IntN(256))
	}

	dstParpar := make([]byte, size)
	gf.MulAdd(dstParpar, src, 0xABCD, scratch)

	dstRef := make([]byte, size)
	refMulAccumulate(dstRef, src, 0xABCD)

	if !bytes.Equal(dstParpar, dstRef) {
		for i := range dstParpar {
			if dstParpar[i] != dstRef[i] {
				t.Errorf("mismatch at byte %d in 768KB buffer", i)
				break
			}
		}
	}
}

// Reference implementation for cross-validation.
func refMulAccumulate(dst, src []byte, factor uint16) {
	if factor == 0 {
		return
	}
	logF := logTable[factor]
	for i := 0; i+1 < len(src); i += 2 {
		val := uint16(src[i]) | uint16(src[i+1])<<8
		if val == 0 {
			continue
		}
		product := expTable[uint32(logTable[val])+uint32(logF)]
		dst[i] ^= byte(product)
		dst[i+1] ^= byte(product >> 8)
	}
}

func BenchmarkMulAdd(b *testing.B) {
	gf, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer gf.Close()

	scratch := gf.NewScratch()
	defer scratch.Free()

	b.Logf("Method: %s", gf.MethodName())

	for _, size := range []int{1024, 4096, 65536, 1 << 20} {
		b.Run(fmt.Sprintf("%dB", size), func(b *testing.B) {
			// Align to stride
			stride := gf.Stride()
			aligned := size - (size % stride)
			if aligned == 0 {
				aligned = stride
			}

			src := make([]byte, aligned)
			dst := make([]byte, aligned)
			for i := range src {
				src[i] = byte(i)
			}

			b.SetBytes(int64(aligned))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				gf.MulAdd(dst, src, 0x1234, scratch)
			}
		})
	}
}

func BenchmarkMulAddScalar(b *testing.B) {
	for _, size := range []int{1024, 4096, 65536, 1 << 20} {
		b.Run(fmt.Sprintf("%dB", size), func(b *testing.B) {
			src := make([]byte, size)
			dst := make([]byte, size)
			for i := range src {
				src[i] = byte(i)
			}

			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				refMulAccumulate(dst, src, 0x1234)
			}
		})
	}
}
