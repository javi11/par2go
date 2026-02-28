package rsenc

import (
	"context"
	"testing"

	"github.com/javi11/par2go/internal/gf16"
)

func TestGenerateConstants(t *testing.T) {
	constants := GenerateConstants(10)
	if len(constants) != 10 {
		t.Fatalf("expected 10 constants, got %d", len(constants))
	}

	// First valid exponent: 1 (1%3!=0, 1%5!=0, 1%17!=0, 1%257!=0)
	// So first constant should be Pow(2, 1) = 2
	if constants[0] != gf16.Pow(2, 1) {
		t.Errorf("first constant: got %d, want %d", constants[0], gf16.Pow(2, 1))
	}

	// Second valid exponent: 2
	if constants[1] != gf16.Pow(2, 2) {
		t.Errorf("second constant: got %d, want %d", constants[1], gf16.Pow(2, 2))
	}

	// Verify skip of 3 (3%3 == 0)
	// Exponents 1, 2, [skip 3], 4, [skip 5], [skip 6], 7, 8, ...
	// 4: 4%3!=0, 4%5!=0, 4%17!=0, 4%257!=0 → valid
	if constants[2] != gf16.Pow(2, 4) {
		t.Errorf("third constant: got %d, want %d (exponent 4)", constants[2], gf16.Pow(2, 4))
	}

	// All constants should be non-zero and distinct
	seen := make(map[uint16]bool)
	for i, c := range constants {
		if c == 0 {
			t.Errorf("constant[%d] is zero", i)
		}
		if seen[c] {
			t.Errorf("duplicate constant at index %d: %d", i, c)
		}
		seen[c] = true
	}
}

func TestGenerateConstantsSkipsCorrectly(t *testing.T) {
	// Verify all skip conditions
	constants := GenerateConstants(100)

	// Regenerate to verify — check that no skipped exponent sneaks in
	n := 0
	for i := 0; i < len(constants); {
		skip := n%3 == 0 || n%5 == 0 || n%17 == 0 || n%257 == 0
		if !skip {
			expected := gf16.Pow(2, uint16(n))
			if constants[i] != expected {
				t.Errorf("constant[%d] (exp=%d): got %d, want %d", i, n, constants[i], expected)
			}
			i++
		}
		n++
	}
}

func TestEncoderBasic(t *testing.T) {
	sliceSize := 64
	numInputSlices := 4
	numRecovery := 2

	enc := NewEncoder(sliceSize, numRecovery)
	enc.SetMemoryBudget(sliceSize * numRecovery * 2) // enough for all at once

	// Create input slices with known data
	inputSlices := make([][]byte, numInputSlices)
	for i := range inputSlices {
		inputSlices[i] = make([]byte, sliceSize)
		for j := 0; j < sliceSize; j += 2 {
			val := uint16((i+1)*100 + j/2)
			inputSlices[i][j] = byte(val)
			inputSlices[i][j+1] = byte(val >> 8)
		}
	}

	// Collect recovery blocks
	recoveryBlocks := make(map[uint16][]byte)

	err := enc.Process(
		context.Background(),
		numInputSlices,
		func(i int) ([]byte, error) { return inputSlices[i], nil },
		nil, // releaseSlice
		func(exponent uint16, data []byte) error {
			buf := make([]byte, len(data))
			copy(buf, data)
			recoveryBlocks[exponent] = buf
			return nil
		},
		gf16.FreeAligned, // releaseRecovery: free aligned buffer after copy
		nil,
	)

	if err != nil {
		t.Fatalf("Process error: %v", err)
	}

	if len(recoveryBlocks) != numRecovery {
		t.Fatalf("expected %d recovery blocks, got %d", numRecovery, len(recoveryBlocks))
	}

	// Verify recovery blocks are non-zero (they should have data)
	for exp, block := range recoveryBlocks {
		allZero := true
		for _, b := range block {
			if b != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("recovery block exp=%d is all zeros", exp)
		}
	}
}

func TestEncoderVerifyManually(t *testing.T) {
	// Verify encoding matches manual computation
	sliceSize := 4 // 2 GF elements
	numInputSlices := 2
	numRecovery := 1

	enc := NewEncoder(sliceSize, numRecovery)

	// Input: slice 0 = [1, 0, 2, 0], slice 1 = [3, 0, 4, 0]
	// (GF elements: [1, 2] and [3, 4])
	inputSlices := [][]byte{
		{1, 0, 2, 0},
		{3, 0, 4, 0},
	}

	constants := GenerateConstants(numInputSlices)

	var recoveryData []byte
	var recoveryExp uint16

	err := enc.Process(
		context.Background(),
		numInputSlices,
		func(i int) ([]byte, error) { return inputSlices[i], nil },
		nil, // releaseSlice
		func(exponent uint16, data []byte) error {
			recoveryData = make([]byte, len(data))
			copy(recoveryData, data)
			recoveryExp = exponent
			return nil
		},
		gf16.FreeAligned, // releaseRecovery
		nil,
	)

	if err != nil {
		t.Fatalf("Process error: %v", err)
	}

	// Manual computation:
	// recovery[0] = sum over i of: inputSlice[i] * (constants[i] ^ exponent[0])
	// exponent[0] = 0, so constants[i]^0 = 1 for all i
	// recovery[0] = inputSlice[0]*1 XOR inputSlice[1]*1

	// With exponent 0, the factor is constant[i]^0 = 1
	expected := make([]byte, sliceSize)
	for i := 0; i < numInputSlices; i++ {
		factor := gf16.Pow(constants[i], recoveryExp)
		gf16.MulAccumulate(expected, inputSlices[i], factor)
	}

	for i, b := range recoveryData {
		if b != expected[i] {
			t.Errorf("byte %d: got %d, want %d", i, b, expected[i])
		}
	}
}

func TestEncoderContextCancel(t *testing.T) {
	sliceSize := 64
	numInputSlices := 100
	numRecovery := 10

	enc := NewEncoder(sliceSize, numRecovery)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := enc.Process(
		ctx,
		numInputSlices,
		func(i int) ([]byte, error) { return make([]byte, sliceSize), nil },
		nil, // releaseSlice
		func(exponent uint16, data []byte) error { return nil },
		gf16.FreeAligned, // releaseRecovery
		nil,
	)

	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

func TestEncoderProgress(t *testing.T) {
	sliceSize := 64
	numInputSlices := 10
	numRecovery := 3

	enc := NewEncoder(sliceSize, numRecovery)

	var lastProgress float64
	progressCalls := 0

	err := enc.Process(
		context.Background(),
		numInputSlices,
		func(i int) ([]byte, error) { return make([]byte, sliceSize), nil },
		nil, // releaseSlice
		func(exponent uint16, data []byte) error { return nil },
		gf16.FreeAligned, // releaseRecovery
		func(pct float64) {
			if pct < lastProgress {
				t.Errorf("progress went backwards: %f -> %f", lastProgress, pct)
			}
			lastProgress = pct
			progressCalls++
		},
	)

	if err != nil {
		t.Fatalf("Process error: %v", err)
	}

	if progressCalls == 0 {
		t.Error("progress callback was never called")
	}
	if lastProgress < 0.99 {
		t.Errorf("final progress %f should be ~1.0", lastProgress)
	}
}

func TestEncoderInputCache(t *testing.T) {
	// Verify that with the input cache enabled, results are byte-for-byte
	// identical to the no-cache path.
	sliceSize := 64
	numInputSlices := 6
	numRecovery := 4

	inputSlices := make([][]byte, numInputSlices)
	for i := range inputSlices {
		inputSlices[i] = make([]byte, sliceSize)
		for j := range inputSlices[i] {
			inputSlices[i][j] = byte(i*13 + j*7)
		}
	}

	readSlice := func(i int) ([]byte, error) { return inputSlices[i], nil }
	collect := func(enc *Encoder) (map[uint16][]byte, error) {
		blocks := make(map[uint16][]byte)
		err := enc.Process(
			context.Background(),
			numInputSlices,
			readSlice,
			nil,
			func(exp uint16, data []byte) error {
				buf := make([]byte, len(data))
				copy(buf, data)
				blocks[exp] = buf
				return nil
			},
			gf16.FreeAligned, // releaseRecovery
			nil,
		)
		return blocks, err
	}

	encNoCache := NewEncoder(sliceSize, numRecovery)
	noCache, err := collect(encNoCache)
	if err != nil {
		t.Fatalf("no-cache Process error: %v", err)
	}

	encWithCache := NewEncoder(sliceSize, numRecovery)
	encWithCache.SetInputCacheBytes(numInputSlices * sliceSize)
	withCache, err := collect(encWithCache)
	if err != nil {
		t.Fatalf("cache Process error: %v", err)
	}

	if len(noCache) != len(withCache) {
		t.Fatalf("block count mismatch: no-cache=%d cache=%d", len(noCache), len(withCache))
	}
	for exp, want := range noCache {
		got, ok := withCache[exp]
		if !ok {
			t.Errorf("missing recovery block exp=%d in cached result", exp)
			continue
		}
		for i, b := range want {
			if got[i] != b {
				t.Errorf("exp=%d byte %d: got %d, want %d", exp, i, got[i], b)
				break
			}
		}
	}
}

func TestEncoderInputCacheMultiBatch(t *testing.T) {
	// Force batching (small MemoryBudget) AND enable the input cache.
	// Results must be identical to the no-cache, no-batch baseline, proving
	// that the cache eliminates repeated reads correctly.
	sliceSize := 64
	numInputSlices := 5
	numRecovery := 8

	inputSlices := make([][]byte, numInputSlices)
	for i := range inputSlices {
		inputSlices[i] = make([]byte, sliceSize)
		for j := range inputSlices[i] {
			inputSlices[i][j] = byte(i*17 + j*3 + 1)
		}
	}

	readCount := 0
	readSlice := func(i int) ([]byte, error) {
		readCount++
		return inputSlices[i], nil
	}
	collect := func(enc *Encoder) (map[uint16][]byte, error) {
		blocks := make(map[uint16][]byte)
		err := enc.Process(
			context.Background(),
			numInputSlices,
			readSlice,
			nil,
			func(exp uint16, data []byte) error {
				buf := make([]byte, len(data))
				copy(buf, data)
				blocks[exp] = buf
				return nil
			},
			gf16.FreeAligned, // releaseRecovery
			nil,
		)
		return blocks, err
	}

	// Baseline: single batch, no cache
	encBase := NewEncoder(sliceSize, numRecovery)
	baseline, err := collect(encBase)
	if err != nil {
		t.Fatalf("baseline Process error: %v", err)
	}

	// Multi-batch + cache: budget = 3 blocks → 3 batches; cache covers all input slices
	readCount = 0
	encCached := NewEncoder(sliceSize, numRecovery)
	encCached.SetMemoryBudget(sliceSize * 3)
	encCached.SetInputCacheBytes(numInputSlices * sliceSize)
	cached, err := collect(encCached)
	if err != nil {
		t.Fatalf("cached+multi-batch Process error: %v", err)
	}

	// With cache enabled, each slice should be read exactly once (during pre-read).
	if readCount != numInputSlices {
		t.Errorf("expected %d reads with cache, got %d", numInputSlices, readCount)
	}

	// Results must match baseline.
	if len(baseline) != len(cached) {
		t.Fatalf("block count mismatch: baseline=%d cached=%d", len(baseline), len(cached))
	}
	for exp, want := range baseline {
		got, ok := cached[exp]
		if !ok {
			t.Errorf("missing recovery block exp=%d in cached result", exp)
			continue
		}
		for i, b := range want {
			if got[i] != b {
				t.Errorf("exp=%d byte %d: got %d, want %d", exp, i, got[i], b)
				break
			}
		}
	}
}

func BenchmarkEncoderProcess(b *testing.B) {
	// Simulate encoding 1MB of data with 10% redundancy
	sliceSize := 10000
	numInputSlices := 100 // ~1MB
	numRecovery := 10

	// Pre-generate input slices
	inputSlices := make([][]byte, numInputSlices)
	for i := range inputSlices {
		inputSlices[i] = make([]byte, sliceSize)
		for j := range inputSlices[i] {
			inputSlices[i][j] = byte(i*sliceSize + j)
		}
	}

	b.SetBytes(int64(sliceSize * numInputSlices))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		enc := NewEncoder(sliceSize, numRecovery)
		_ = enc.Process(
			context.Background(),
			numInputSlices,
			func(idx int) ([]byte, error) { return inputSlices[idx], nil },
			nil, // releaseSlice
			func(exponent uint16, data []byte) error { return nil },
			gf16.FreeAligned, // releaseRecovery
			nil,
		)
	}
}

func TestEncoderBatching(t *testing.T) {
	// Force batching by setting a small memory budget
	sliceSize := 64
	numInputSlices := 4
	numRecovery := 8

	enc := NewEncoder(sliceSize, numRecovery)
	enc.SetMemoryBudget(sliceSize * 3) // Only 3 blocks fit at once → multiple batches

	inputSlices := make([][]byte, numInputSlices)
	for i := range inputSlices {
		inputSlices[i] = make([]byte, sliceSize)
		for j := range inputSlices[i] {
			inputSlices[i][j] = byte(i*sliceSize + j)
		}
	}

	recoveryBlocks := make(map[uint16][]byte)

	err := enc.Process(
		context.Background(),
		numInputSlices,
		func(i int) ([]byte, error) { return inputSlices[i], nil },
		nil, // releaseSlice
		func(exponent uint16, data []byte) error {
			buf := make([]byte, len(data))
			copy(buf, data)
			recoveryBlocks[exponent] = buf
			return nil
		},
		gf16.FreeAligned, // releaseRecovery
		nil,
	)

	if err != nil {
		t.Fatalf("Process error: %v", err)
	}

	if len(recoveryBlocks) != numRecovery {
		t.Fatalf("expected %d recovery blocks, got %d", numRecovery, len(recoveryBlocks))
	}
}

func TestEncoderZeroCopyRecovery(t *testing.T) {
	// Verify that releaseRecovery is called exactly numRecovery times and that
	// the data passed to writeRecovery matches the baseline (copy-path) output.
	sliceSize := 64
	numInputSlices := 4
	numRecovery := 5

	enc := NewEncoder(sliceSize, numRecovery)
	enc.SetMemoryBudget(sliceSize * numRecovery) // single batch

	inputSlices := make([][]byte, numInputSlices)
	for i := range inputSlices {
		inputSlices[i] = make([]byte, sliceSize)
		for j := range inputSlices[i] {
			inputSlices[i][j] = byte(i*13 + j*7)
		}
	}

	// Baseline: copy path
	baselineBlocks := make(map[uint16][]byte)
	if err := enc.Process(
		context.Background(),
		numInputSlices,
		func(i int) ([]byte, error) { return inputSlices[i], nil },
		nil,
		func(exp uint16, data []byte) error {
			buf := make([]byte, len(data))
			copy(buf, data)
			baselineBlocks[exp] = buf
			return nil
		},
		gf16.FreeAligned,
		nil,
	); err != nil {
		t.Fatalf("baseline Process error: %v", err)
	}

	// Zero-copy path: count releaseRecovery calls, verify data equals baseline
	releaseCount := 0
	copiedBlocks := make(map[uint16][]byte)
	if err := enc.Process(
		context.Background(),
		numInputSlices,
		func(i int) ([]byte, error) { return inputSlices[i], nil },
		nil,
		func(exp uint16, data []byte) error {
			buf := make([]byte, len(data))
			copy(buf, data) // copy before release
			copiedBlocks[exp] = buf
			return nil
		},
		func(data []byte) {
			releaseCount++
			gf16.FreeAligned(data)
		},
		nil,
	); err != nil {
		t.Fatalf("zero-copy Process error: %v", err)
	}

	if releaseCount != numRecovery {
		t.Errorf("releaseRecovery called %d times, want %d", releaseCount, numRecovery)
	}

	if len(copiedBlocks) != numRecovery {
		t.Fatalf("expected %d blocks, got %d", numRecovery, len(copiedBlocks))
	}
	for exp, want := range baselineBlocks {
		got, ok := copiedBlocks[exp]
		if !ok {
			t.Errorf("missing block exp=%d", exp)
			continue
		}
		for i, b := range want {
			if got[i] != b {
				t.Errorf("exp=%d byte %d: got %d, want %d", exp, i, got[i], b)
				break
			}
		}
	}
}
