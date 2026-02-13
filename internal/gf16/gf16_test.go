package gf16

import (
	"testing"
)

func TestAdd(t *testing.T) {
	// Addition is XOR in GF(2^k)
	if Add(0, 0) != 0 {
		t.Error("0 + 0 should be 0")
	}
	if Add(1, 0) != 1 {
		t.Error("1 + 0 should be 1")
	}
	if Add(0xFFFF, 0xFFFF) != 0 {
		t.Error("a + a should be 0 in GF(2^16)")
	}
	if Add(0x1234, 0x5678) != 0x1234^0x5678 {
		t.Error("Add should be XOR")
	}
}

func TestMul(t *testing.T) {
	// Multiplication by 0
	if Mul(0, 12345) != 0 {
		t.Error("0 * x should be 0")
	}
	if Mul(12345, 0) != 0 {
		t.Error("x * 0 should be 0")
	}

	// Multiplication by 1
	if Mul(1, 12345) != 12345 {
		t.Error("1 * x should be x")
	}
	if Mul(12345, 1) != 12345 {
		t.Error("x * 1 should be x")
	}

	// Commutativity
	a, b := uint16(1234), uint16(5678)
	if Mul(a, b) != Mul(b, a) {
		t.Errorf("Mul(%d, %d) != Mul(%d, %d)", a, b, b, a)
	}

	// a * a^(-1) = 1
	for _, val := range []uint16{1, 2, 3, 100, 1000, 65535} {
		inv := Inv(val)
		product := Mul(val, inv)
		if product != 1 {
			t.Errorf("Mul(%d, Inv(%d)) = %d, want 1", val, val, product)
		}
	}
}

func TestPow(t *testing.T) {
	// x^0 = 1
	if Pow(12345, 0) != 1 {
		t.Error("x^0 should be 1")
	}

	// x^1 = x
	if Pow(12345, 1) != 12345 {
		t.Error("x^1 should be x")
	}

	// 0^n = 0 for n > 0
	if Pow(0, 5) != 0 {
		t.Error("0^n should be 0 for n > 0")
	}

	// 2^16 in GF(2^16) — known value with our polynomial
	// The generator is 2, so exp_table[16] = 2^16 mod polynomial
	expected := expTable[16]
	if Pow(2, 16) != expected {
		t.Errorf("Pow(2, 16) = %d, want %d", Pow(2, 16), expected)
	}

	// Verify Pow via repeated Mul
	base := uint16(7)
	result := uint16(1)
	for i := 0; i < 10; i++ {
		if Pow(base, uint16(i)) != result {
			t.Errorf("Pow(%d, %d) = %d, want %d", base, i, Pow(base, uint16(i)), result)
		}
		result = Mul(result, base)
	}
}

func TestInv(t *testing.T) {
	// Inv(1) = 1
	if Inv(1) != 1 {
		t.Error("Inv(1) should be 1")
	}

	// Test a range of values
	for v := uint16(1); v <= 1000; v++ {
		inv := Inv(v)
		if Mul(v, inv) != 1 {
			t.Errorf("v=%d: Mul(v, Inv(v)) = %d, want 1", v, Mul(v, inv))
		}
	}
}

func TestExpLogConsistency(t *testing.T) {
	// For all non-zero elements: expTable[logTable[x]] == x
	for x := uint16(1); x != 0; x++ {
		if expTable[logTable[x]] != x {
			t.Errorf("exp(log(%d)) = %d", x, expTable[logTable[x]])
			break
		}
	}
}

func TestMulAccumulate(t *testing.T) {
	// Test basic multiply-accumulate
	src := make([]byte, 64)
	dst := make([]byte, 64)
	expected := make([]byte, 64)

	// Fill src with test pattern (little-endian uint16 values)
	for i := 0; i < len(src); i += 2 {
		val := uint16(i/2 + 1)
		src[i] = byte(val)
		src[i+1] = byte(val >> 8)
	}

	factor := uint16(42)

	// Compute expected result manually
	for i := 0; i < len(src); i += 2 {
		val := uint16(src[i]) | uint16(src[i+1])<<8
		product := Mul(val, factor)
		expected[i] = byte(product)
		expected[i+1] = byte(product >> 8)
	}

	MulAccumulate(dst, src, factor)

	for i := range dst {
		if dst[i] != expected[i] {
			t.Errorf("byte %d: got %d, want %d", i, dst[i], expected[i])
		}
	}
}

func TestMulAccumulateXOR(t *testing.T) {
	// Test that MulAccumulate XORs into dst (not overwrites)
	src := make([]byte, 16)
	dst := make([]byte, 16)

	// Set src to [1, 0, 2, 0, 3, 0, ...] (little-endian uint16)
	for i := 0; i < len(src); i += 2 {
		val := uint16(i/2 + 1)
		src[i] = byte(val)
		src[i+1] = byte(val >> 8)
	}

	// Pre-fill dst
	for i := range dst {
		dst[i] = 0xFF
	}
	dstCopy := make([]byte, len(dst))
	copy(dstCopy, dst)

	factor := uint16(5)
	MulAccumulate(dst, src, factor)

	// Verify it XORed, not overwrote
	for i := 0; i < len(src); i += 2 {
		val := uint16(src[i]) | uint16(src[i+1])<<8
		product := Mul(val, factor)
		expectedLo := dstCopy[i] ^ byte(product)
		expectedHi := dstCopy[i+1] ^ byte(product>>8)
		if dst[i] != expectedLo || dst[i+1] != expectedHi {
			t.Errorf("pos %d: got [%d,%d], want [%d,%d]", i, dst[i], dst[i+1], expectedLo, expectedHi)
		}
	}
}

func TestMulAccumulateFactorZero(t *testing.T) {
	src := []byte{1, 2, 3, 4}
	dst := []byte{0xFF, 0xFF, 0xFF, 0xFF}
	MulAccumulate(dst, src, 0)
	// dst should be unchanged
	for i, v := range dst {
		if v != 0xFF {
			t.Errorf("byte %d: got %d, want 0xFF", i, v)
		}
	}
}

func TestMulAccumulateFactorOne(t *testing.T) {
	src := make([]byte, 16)
	dst := make([]byte, 16)

	for i := range src {
		src[i] = byte(i + 1)
	}

	MulAccumulate(dst, src, 1)

	// Multiply by 1 should just XOR src into dst
	for i := range dst {
		if dst[i] != src[i] {
			t.Errorf("byte %d: got %d, want %d", i, dst[i], src[i])
		}
	}
}

func BenchmarkMul(b *testing.B) {
	a, bb := uint16(12345), uint16(54321)
	for i := 0; i < b.N; i++ {
		_ = Mul(a, bb)
	}
}

func BenchmarkMulAccumulate(b *testing.B) {
	sizes := []int{1024, 4096, 65536, 1 << 20}
	for _, size := range sizes {
		b.Run(byteSizeName(size), func(b *testing.B) {
			src := make([]byte, size)
			dst := make([]byte, size)
			for i := range src {
				src[i] = byte(i)
			}
			factor := uint16(42)
			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MulAccumulate(dst, src, factor)
			}
		})
	}
}

func BenchmarkMulAccumulateScalar(b *testing.B) {
	// Benchmark the pure Go scalar path for comparison
	sizes := []int{1024, 4096, 65536, 1 << 20}
	for _, size := range sizes {
		b.Run(byteSizeName(size), func(b *testing.B) {
			src := make([]byte, size)
			dst := make([]byte, size)
			for i := range src {
				src[i] = byte(i)
			}
			factor := uint16(42)
			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				mulAccumulate(dst, src, factor)
			}
		})
	}
}

func TestMulAccumulateLargeBuffer(t *testing.T) {
	// Test with a PAR2-realistic buffer size (768KB slice size)
	size := 768 * 1024
	src := make([]byte, size)
	dst := make([]byte, size)
	expected := make([]byte, size)

	for i := 0; i < len(src); i += 2 {
		val := uint16(i/2 + 1)
		src[i] = byte(val)
		src[i+1] = byte(val >> 8)
	}

	factor := uint16(12345)

	// Compute expected via scalar
	for i := 0; i < len(src); i += 2 {
		val := uint16(src[i]) | uint16(src[i+1])<<8
		product := Mul(val, factor)
		expected[i] = byte(product)
		expected[i+1] = byte(product >> 8)
	}

	MulAccumulate(dst, src, factor)

	for i := range dst {
		if dst[i] != expected[i] {
			t.Errorf("byte %d: got %d, want %d", i, dst[i], expected[i])
			break
		}
	}
}

func TestMulAccumulateOddSizes(t *testing.T) {
	// Test sizes that aren't multiples of 16 or 32 to exercise tail handling
	for _, size := range []int{2, 6, 14, 18, 30, 34, 50} {
		t.Run(byteSizeName(size), func(t *testing.T) {
			src := make([]byte, size)
			dst := make([]byte, size)
			expected := make([]byte, size)

			for i := 0; i < size; i += 2 {
				val := uint16(i/2 + 7)
				src[i] = byte(val)
				src[i+1] = byte(val >> 8)
			}

			factor := uint16(999)
			for i := 0; i < size; i += 2 {
				val := uint16(src[i]) | uint16(src[i+1])<<8
				product := Mul(val, factor)
				expected[i] = byte(product)
				expected[i+1] = byte(product >> 8)
			}

			MulAccumulate(dst, src, factor)

			for i := range dst {
				if dst[i] != expected[i] {
					t.Errorf("size=%d byte %d: got %d, want %d", size, i, dst[i], expected[i])
					break
				}
			}
		})
	}
}

func byteSizeName(n int) string {
	switch {
	case n >= 1<<20:
		return string(rune('0'+n/(1<<20))) + "MB"
	case n >= 1<<10:
		return string(rune('0'+n/(1<<10))) + "KB"
	default:
		return string(rune('0'+n)) + "B"
	}
}
