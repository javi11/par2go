#include "textflag.h"

// Shuffle patterns for deinterleaving little-endian uint16 pairs.
// even: extracts bytes at positions 0,2,4,6,8,10,12,14 (lo bytes of each uint16)
// odd:  extracts bytes at positions 1,3,5,7,9,11,13,15 (hi bytes of each uint16)
// 0x80 causes PSHUFB to output zero (clears positions 8-15).
DATA even_shuffle<>+0(SB)/8, $0x0E0C0A0806040200
DATA even_shuffle<>+8(SB)/8, $0x8080808080808080
GLOBL even_shuffle<>(SB), RODATA|NOPTR, $16

DATA odd_shuffle<>+0(SB)/8, $0x0F0D0B0907050301
DATA odd_shuffle<>+8(SB)/8, $0x8080808080808080
GLOBL odd_shuffle<>(SB), RODATA|NOPTR, $16

// func mulAccumulateSSSE3(dst, src []byte, tables *MulAccTables)
// Processes floor(len(src)/16)*16 bytes using SSSE3 PSHUFB.
// Each 16 input bytes = 8 GF(2^16) elements processed in parallel.
//
// Algorithm:
// 1. Deinterleave [lo0,hi0,lo1,hi1,...] into separate lo/hi byte vectors
// 2. Extract 4 nibble planes from the lo and hi bytes
// 3. PSHUFB lookup through 8 precomputed tables, XOR partial results
// 4. PUNPCKLBW to interleave result_lo and result_hi back
// 5. XOR into dst
TEXT ·mulAccumulateSSSE3(SB), NOSPLIT, $0-56
	MOVQ dst_base+0(FP), DI
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), CX
	MOVQ tables+48(FP), DX

	ANDQ $~15, CX
	JZ   ssse3_done

	// Load constants
	MOVQ $0x0F0F0F0F0F0F0F0F, AX
	MOVQ AX, X15
	PUNPCKLQDQ X15, X15            // X15 = nibble mask 0x0F

	MOVOU even_shuffle<>(SB), X14  // X14 = deinterleave even bytes
	MOVOU odd_shuffle<>(SB), X13   // X13 = deinterleave odd bytes

	// Load tables 0-4 into registers
	MOVOU 0(DX), X8               // table[0]
	MOVOU 16(DX), X9              // table[1]
	MOVOU 32(DX), X10             // table[2]
	MOVOU 48(DX), X11             // table[3]
	MOVOU 64(DX), X12             // table[4]

ssse3_loop:
	// Load 16 input bytes: [lo0,hi0,lo1,hi1,...,lo7,hi7]
	MOVOU (SI), X0

	// Deinterleave into lo bytes and hi bytes
	MOVO X0, X1
	PSHUFB X14, X0                 // X0 = [lo0,lo1,...,lo7, 0,0,...,0]
	PSHUFB X13, X1                 // X1 = [hi0,hi1,...,hi7, 0,0,...,0]

	// Extract 4 nibble planes
	MOVO X0, X2
	PSRLW $4, X2
	PAND X15, X0                   // X0 = nib0 (low nibble of lo bytes)
	PAND X15, X2                   // X2 = nib1 (high nibble of lo bytes)
	MOVO X1, X3
	PSRLW $4, X3
	PAND X15, X1                   // X1 = nib2 (low nibble of hi bytes)
	PAND X15, X3                   // X3 = nib3 (high nibble of hi bytes)

	// === result_lo = tab0[nib0] ^ tab1[nib1] ^ tab2[nib2] ^ tab3[nib3] ===
	MOVO X8, X4
	PSHUFB X0, X4                  // tab0[nib0]
	MOVO X9, X5
	PSHUFB X2, X5                  // tab1[nib1]
	PXOR X5, X4
	MOVO X10, X5
	PSHUFB X1, X5                  // tab2[nib2]
	PXOR X5, X4
	MOVO X11, X5
	PSHUFB X3, X5                  // tab3[nib3]
	PXOR X5, X4                    // X4 = result_lo (8 valid bytes in 0-7)

	// === result_hi = tab4[nib0] ^ tab5[nib1] ^ tab6[nib2] ^ tab7[nib3] ===
	MOVO X12, X5
	PSHUFB X0, X5                  // tab4[nib0]
	MOVOU 80(DX), X6
	PSHUFB X2, X6                  // tab5[nib1]
	PXOR X6, X5
	MOVOU 96(DX), X6
	PSHUFB X1, X6                  // tab6[nib2]
	PXOR X6, X5
	MOVOU 112(DX), X6
	PSHUFB X3, X6                  // tab7[nib3]
	PXOR X6, X5                    // X5 = result_hi (8 valid bytes in 0-7)

	// Interleave result_lo and result_hi back to uint16 layout
	PUNPCKLBW X5, X4               // X4 = [rlo0,rhi0,rlo1,rhi1,...,rlo7,rhi7]

	// XOR into dst
	MOVOU (DI), X0
	PXOR X4, X0
	MOVOU X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $16, CX
	JNZ  ssse3_loop

ssse3_done:
	RET

// func mulAccumulateAVX2(dst, src []byte, tables *MulAccTables)
// Processes floor(len(src)/32)*32 bytes using AVX2 VPSHUFB.
// Each 32 input bytes = 16 GF(2^16) elements processed in parallel.
// VPSHUFB operates independently on each 128-bit lane, so the
// deinterleave/process/interleave works identically per lane.
TEXT ·mulAccumulateAVX2(SB), NOSPLIT, $0-56
	MOVQ dst_base+0(FP), DI
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), CX
	MOVQ tables+48(FP), DX

	ANDQ $~31, CX
	JZ   avx2_done

	// Broadcast nibble mask to all 32 bytes
	MOVQ $0x0F0F0F0F0F0F0F0F, AX
	MOVQ AX, X15
	VPBROADCASTQ X15, Y15

	// Broadcast shuffle patterns to both lanes
	VBROADCASTI128 even_shuffle<>(SB), Y14
	VBROADCASTI128 odd_shuffle<>(SB), Y13

	// Load and broadcast tables 0-4
	VBROADCASTI128 0(DX), Y8
	VBROADCASTI128 16(DX), Y9
	VBROADCASTI128 32(DX), Y10
	VBROADCASTI128 48(DX), Y11
	VBROADCASTI128 64(DX), Y12

avx2_loop:
	// Load 32 input bytes
	VMOVDQU (SI), Y0

	// Deinterleave per lane
	VPSHUFB Y14, Y0, Y1            // Y1 = lo bytes deinterleaved per lane
	VPSHUFB Y13, Y0, Y2            // Y2 = hi bytes deinterleaved per lane

	// Extract nibbles
	VPSRLW $4, Y1, Y3              // Y3 = high nibbles of lo bytes
	VPSRLW $4, Y2, Y4              // Y4 = high nibbles of hi bytes
	VPAND Y15, Y1, Y0              // Y0 = nib0 (low nibble of lo bytes)
	VPAND Y15, Y3, Y1              // Y1 = nib1 (high nibble of lo bytes)
	VPAND Y15, Y2, Y2              // Y2 = nib2 (low nibble of hi bytes)
	VPAND Y15, Y4, Y3              // Y3 = nib3 (high nibble of hi bytes)

	// result_lo = tab0[nib0] ^ tab1[nib1] ^ tab2[nib2] ^ tab3[nib3]
	VPSHUFB Y0, Y8, Y4             // tab0[nib0]
	VPSHUFB Y1, Y9, Y5             // tab1[nib1]
	VPXOR Y5, Y4, Y4
	VPSHUFB Y2, Y10, Y5            // tab2[nib2]
	VPXOR Y5, Y4, Y4
	VPSHUFB Y3, Y11, Y5            // tab3[nib3]
	VPXOR Y5, Y4, Y4               // Y4 = result_lo

	// result_hi = tab4[nib0] ^ tab5[nib1] ^ tab6[nib2] ^ tab7[nib3]
	VPSHUFB Y0, Y12, Y5            // tab4[nib0]
	VBROADCASTI128 80(DX), Y6
	VPSHUFB Y1, Y6, Y6             // tab5[nib1]
	VPXOR Y6, Y5, Y5
	VBROADCASTI128 96(DX), Y6
	VPSHUFB Y2, Y6, Y6             // tab6[nib2]
	VPXOR Y6, Y5, Y5
	VBROADCASTI128 112(DX), Y6
	VPSHUFB Y3, Y6, Y6             // tab7[nib3]
	VPXOR Y6, Y5, Y5               // Y5 = result_hi

	// Interleave per lane
	VPUNPCKLBW Y5, Y4, Y4          // Y4 = interleaved result

	// XOR into dst
	VMOVDQU (DI), Y0
	VPXOR Y4, Y0, Y0
	VMOVDQU Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JNZ  avx2_loop

	VZEROUPPER

avx2_done:
	RET
