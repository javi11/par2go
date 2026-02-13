#include "textflag.h"

// func mulAccumulateNEON(dst, src []byte, tables *MulAccTables)
// Processes floor(len(src)/16)*16 bytes using NEON VTBL (TBL).
// Each 16 input bytes = 8 GF(2^16) elements processed in parallel.
//
// Algorithm: deinterleave + split-table + interleave.
// 1. VTBL deinterleave [lo0,hi0,...] → lo bytes + hi bytes
// 2. Extract 4 nibble planes, VTBL lookup through 8 tables
// 3. XOR partial results for product_lo and product_hi
// 4. VTBL interleave back, XOR into dst
TEXT ·mulAccumulateNEON(SB), NOSPLIT, $0-56
	MOVD dst_base+0(FP), R0       // dst pointer
	MOVD src_base+24(FP), R1      // src pointer
	MOVD src_len+32(FP), R2       // length
	MOVD tables+48(FP), R3        // tables pointer

	// Round down to multiple of 16
	AND  $~15, R2
	CBZ  R2, neon_done

	// Load nibble mask 0x0F into V31.B16
	VMOVI $15, V31.B16

	// Load all 8 tables into V16-V23
	VLD1 (R3), [V16.B16, V17.B16, V18.B16, V19.B16]
	ADD  $64, R3
	VLD1 (R3), [V20.B16, V21.B16, V22.B16, V23.B16]

	// Build deinterleave shuffle patterns
	// even_shuf: {0, 2, 4, 6, 8, 10, 12, 14, 0xFF, 0xFF, ...}
	// odd_shuf:  {1, 3, 5, 7, 9, 11, 13, 15, 0xFF, 0xFF, ...}
	// For NEON TBL, out-of-range indices (>= 16) produce 0.

	// Build even_shuf in V28: 0,2,4,6,8,10,12,14,255,255,...
	MOVD $0x0E0C0A0806040200, R4
	MOVD $0xFFFFFFFFFFFFFFFF, R5
	VMOV R4, V28.D[0]
	VMOV R5, V28.D[1]

	// Build odd_shuf in V29: 1,3,5,7,9,11,13,15,255,255,...
	MOVD $0x0F0D0B0907050301, R4
	VMOV R4, V29.D[0]
	VMOV R5, V29.D[1]

	// Build interleave pattern in V30: {0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15}
	// This takes result_lo[0..7] from positions 0-7 and result_hi[0..7] from positions 8-15
	// and interleaves them: [lo0,hi0,lo1,hi1,...]
	MOVD $0x0B030A0209010800, R4
	VMOV R4, V30.D[0]
	MOVD $0x0F070E060D050C04, R4
	VMOV R4, V30.D[1]

neon_loop:
	// Load 16 src bytes (8 GF elements in LE)
	VLD1.P 16(R1), [V0.B16]

	// Deinterleave: V1 = lo bytes, V2 = hi bytes
	VTBL V28.B16, [V0.B16], V1.B16   // V1 = [lo0,lo1,...,lo7, 0,0,...,0]
	VTBL V29.B16, [V0.B16], V2.B16   // V2 = [hi0,hi1,...,hi7, 0,0,...,0]

	// Extract 4 nibble planes
	VAND V31.B16, V1.B16, V3.B16     // V3 = nib0 (low nibble of lo bytes)
	VUSHR $4, V1.B16, V4.B16         // V4 = nib1 (high nibble of lo bytes)
	VAND V31.B16, V2.B16, V5.B16     // V5 = nib2 (low nibble of hi bytes)
	VUSHR $4, V2.B16, V6.B16         // V6 = nib3 (high nibble of hi bytes)

	// result_lo = tab0[nib0] ^ tab1[nib1] ^ tab2[nib2] ^ tab3[nib3]
	VTBL V3.B16, [V16.B16], V7.B16   // tab0[nib0]
	VTBL V4.B16, [V17.B16], V8.B16   // tab1[nib1]
	VEOR V8.B16, V7.B16, V7.B16
	VTBL V5.B16, [V18.B16], V8.B16   // tab2[nib2]
	VEOR V8.B16, V7.B16, V7.B16
	VTBL V6.B16, [V19.B16], V8.B16   // tab3[nib3]
	VEOR V8.B16, V7.B16, V7.B16      // V7 = result_lo (bytes 0-7 valid)

	// result_hi = tab4[nib0] ^ tab5[nib1] ^ tab6[nib2] ^ tab7[nib3]
	VTBL V3.B16, [V20.B16], V8.B16   // tab4[nib0]
	VTBL V4.B16, [V21.B16], V9.B16   // tab5[nib1]
	VEOR V9.B16, V8.B16, V8.B16
	VTBL V5.B16, [V22.B16], V9.B16   // tab6[nib2]
	VEOR V9.B16, V8.B16, V8.B16
	VTBL V6.B16, [V23.B16], V9.B16   // tab7[nib3]
	VEOR V9.B16, V8.B16, V8.B16      // V8 = result_hi (bytes 0-7 valid)

	// Combine result_lo (V7, bytes 0-7) and result_hi (V8, bytes 0-7)
	// into V9 with interleaved layout: [lo0,hi0,lo1,hi1,...]
	// Move result_hi to bytes 8-15 of a combined register, then shuffle
	// V7 = [rlo0,...,rlo7, garbage]
	// V8 = [rhi0,...,rhi7, garbage]
	// Copy V8 bytes 0-7 into V7 bytes 8-15
	VMOV V8.D[0], V7.D[1]
	// Now V7 = [rlo0,...,rlo7, rhi0,...,rhi7]
	// Interleave with V30 shuffle
	VTBL V30.B16, [V7.B16], V9.B16   // V9 = [rlo0,rhi0,rlo1,rhi1,...,rlo7,rhi7]

	// XOR into dst
	VLD1 (R0), [V0.B16]
	VEOR V9.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R0)

	SUB  $16, R2
	CBNZ R2, neon_loop

neon_done:
	RET
