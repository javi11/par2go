//go:build amd64

package gf16

import "golang.org/x/sys/cpu"

var useSIMD = cpu.X86.HasSSSE3
var useAVX2 = cpu.X86.HasAVX2

// mulAccumulate is the pure Go fallback for when SIMD is not used.
func mulAccumulate(dst, src []byte, factor uint16) {
	logFactor := logTable[factor]
	n := len(src)

	for i := 0; i < n; i += 2 {
		val := uint16(src[i]) | uint16(src[i+1])<<8
		if val == 0 {
			continue
		}
		product := expTable[uint32(logTable[val])+uint32(logFactor)]
		dst[i] ^= byte(product)
		dst[i+1] ^= byte(product >> 8)
	}
}

//go:noescape
func mulAccumulateSSSE3(dst, src []byte, tables *MulAccTables)

//go:noescape
func mulAccumulateAVX2(dst, src []byte, tables *MulAccTables)

// mulAccTail handles remaining bytes (< 16) using the precomputed tables.
func mulAccTail(dst, src []byte, tables *MulAccTables) {
	for i := 0; i < len(src); i += 2 {
		lo := src[i]
		hi := src[i+1]
		n0 := lo & 0x0F
		n1 := lo >> 4
		n2 := hi & 0x0F
		n3 := hi >> 4
		dst[i] ^= tables[0][n0] ^ tables[1][n1] ^ tables[2][n2] ^ tables[3][n3]
		dst[i+1] ^= tables[4][n0] ^ tables[5][n1] ^ tables[6][n2] ^ tables[7][n3]
	}
}

func mulAccumulateSIMD(dst, src []byte, tables *MulAccTables) {
	n := len(src)

	if useAVX2 && n >= 32 {
		// AVX2 processes 32 bytes at a time; asm handles n&^31 bytes
		mulAccumulateAVX2(dst, src, tables)
		tail := n & 31
		if tail > 0 {
			off := n - tail
			mulAccTail(dst[off:], src[off:], tables)
		}
		return
	}

	if n >= 16 {
		// SSSE3 processes 16 bytes at a time; asm handles n&^15 bytes
		mulAccumulateSSSE3(dst, src, tables)
		tail := n & 15
		if tail > 0 {
			off := n - tail
			mulAccTail(dst[off:], src[off:], tables)
		}
		return
	}

	// Very small buffers: pure table-based scalar
	mulAccTail(dst, src, tables)
}
