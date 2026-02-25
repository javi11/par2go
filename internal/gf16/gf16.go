// Package gf16 implements GF(2^16) arithmetic as required by the PAR2 specification.
//
// The field is GF(2^16) with the irreducible polynomial x^16 + x^12 + x^3 + x + 1
// (0x1100B in hex). All arithmetic operations are performed in this field.
//
// The hot path — MulAccumulate — dispatches to ParPar's SIMD-optimized backends
// (SSE2, AVX2, AVX-512, NEON, SVE2, etc.) via CGO with runtime CPU detection.
package gf16

// Field polynomial: x^16 + x^12 + x^3 + x + 1 = 0x1100B
const polynomial = 0x1100B

// generator is the primitive element used to build log/exp tables.
// 2 is a generator of GF(2^16) with polynomial 0x1100B.
const generator = 2

var (
	// logTable maps field elements to their discrete logarithm (base generator).
	// logTable[0] is undefined (log of 0 doesn't exist).
	logTable [65536]uint16

	// expTable maps discrete logarithms to field elements.
	// expTable[i] = generator^i mod polynomial.
	// We store 2*65535 entries to avoid modular reduction during Mul.
	expTable [2 * 65535]uint16
)

func init() {
	buildTables()
}

// buildTables computes the log and exp tables for GF(2^16).
func buildTables() {
	_ = generator // document that tables use this generator
	var val uint32 = 1
	for i := 0; i < 65535; i++ {
		expTable[i] = uint16(val)
		expTable[i+65535] = uint16(val) // duplicate for wraparound
		logTable[val] = uint16(i)

		// Multiply by generator in GF(2^16)
		val <<= 1
		if val&0x10000 != 0 {
			val ^= polynomial
		}
	}
	// logTable[0] is left as 0; callers must check for zero.
	// logTable[1] = 0 which is correct (generator^0 = 1).
}

// Add returns a + b in GF(2^16). Addition in GF(2^k) is XOR.
func Add(a, b uint16) uint16 {
	return a ^ b
}

// Mul returns a * b in GF(2^16) using log/exp tables.
func Mul(a, b uint16) uint16 {
	if a == 0 || b == 0 {
		return 0
	}
	return expTable[uint32(logTable[a])+uint32(logTable[b])]
}

// Pow returns base^exp in GF(2^16) using repeated squaring.
func Pow(base, exp uint16) uint16 {
	if exp == 0 {
		return 1
	}
	if base == 0 {
		return 0
	}
	// Use log/exp: base^exp = exp_table[(log(base) * exp) mod 65535]
	logBase := uint32(logTable[base])
	logResult := (logBase * uint32(exp)) % 65535
	return expTable[logResult]
}

// Inv returns the multiplicative inverse of a in GF(2^16).
// Panics if a == 0 (zero has no inverse).
func Inv(a uint16) uint16 {
	if a == 0 {
		panic("gf16: inverse of zero")
	}
	// a^(-1) = exp_table[65535 - log(a)]
	return expTable[65535-uint32(logTable[a])]
}

// MulAccumulate computes dst[i] ^= src[i] * factor for all i, where
// multiplication is in GF(2^16). src and dst are treated as slices of
// little-endian uint16 values, so len must be even.
//
// This is the hot path for RS encoding. It dispatches to ParPar's
// SIMD-optimized backends via CGO.
func MulAccumulate(dst, src []byte, factor uint16) {
	if factor == 0 {
		return
	}
	if len(src) != len(dst) {
		panic("gf16: MulAccumulate src and dst length mismatch")
	}
	if len(src)%2 != 0 {
		panic("gf16: MulAccumulate length must be even")
	}
	if len(src) == 0 {
		return
	}

	if factor == 1 {
		// Multiply by 1 is just XOR
		xorBytes(dst, src)
		return
	}

	mulAccumulate(dst, src, factor)
}

// xorBytes XORs src into dst.
func xorBytes(dst, src []byte) {
	// Process 8 bytes at a time
	n := len(src)
	i := 0
	for ; i+8 <= n; i += 8 {
		dst[i] ^= src[i]
		dst[i+1] ^= src[i+1]
		dst[i+2] ^= src[i+2]
		dst[i+3] ^= src[i+3]
		dst[i+4] ^= src[i+4]
		dst[i+5] ^= src[i+5]
		dst[i+6] ^= src[i+6]
		dst[i+7] ^= src[i+7]
	}
	for ; i < n; i++ {
		dst[i] ^= src[i]
	}
}
