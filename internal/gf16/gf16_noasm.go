//go:build !(amd64 || arm64)

package gf16

var useSIMD = false

// mulAccumulateSIMD is a no-op on unsupported platforms; dispatch never reaches here
// because useSIMD is false, but we need it to satisfy the compiler.
func mulAccumulateSIMD(_, _ []byte, _ *MulAccTables) {}

// mulAccumulate is the pure Go fallback for MulAccumulate.
// It processes data as little-endian uint16 pairs using log/exp table lookups.
func mulAccumulate(dst, src []byte, factor uint16) {
	logFactor := logTable[factor]
	n := len(src)

	for i := 0; i < n; i += 2 {
		// Read little-endian uint16 from src
		val := uint16(src[i]) | uint16(src[i+1])<<8
		if val == 0 {
			continue
		}
		// GF multiply via log/exp
		product := expTable[uint32(logTable[val])+uint32(logFactor)]
		// XOR into dst (little-endian)
		dst[i] ^= byte(product)
		dst[i+1] ^= byte(product >> 8)
	}
}
