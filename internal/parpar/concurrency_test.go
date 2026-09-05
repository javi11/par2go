package parpar

import (
	"bytes"
	"sync"
	"testing"
)

// encodeAll feeds numInputs deterministic slices through a fresh GfProc and
// returns the recovery blocks. When workers > 1 the slices are submitted from
// that many goroutines at once, the way the creator's per-file readers do.
func encodeAll(t *testing.T, sliceSize, numInputs, numRecovery, workers int) [][]byte {
	t.Helper()
	proc, err := NewGfProcWithConfig(GfProcConfig{SliceSize: sliceSize, NumThreads: 2, StagingAreas: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()

	exps := make([]uint16, numRecovery)
	for i := range exps {
		exps[i] = uint16(i)
	}
	proc.SetRecoverySlices(exps)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w; i < numInputs; i += workers {
				data := make([]byte, sliceSize)
				for j := range data {
					data[j] = byte(i*13 + j*3)
				}
				proc.Add(i, data)
			}
		}(w)
	}
	wg.Wait()
	proc.End()

	results := make([][]byte, numRecovery)
	for i := range results {
		results[i] = make([]byte, sliceSize)
		proc.GetOutput(i, results[i])
	}
	return results
}

// TestGfProcAdd_ConcurrentCallersMatchSequential guards the documented
// contract that Add may be called from multiple goroutines: the recovery
// data must be identical to a single-goroutine submission of the same slices.
func TestGfProcAdd_ConcurrentCallersMatchSequential(t *testing.T) {
	const sliceSize, numInputs, numRecovery = 4096, 400, 3

	want := encodeAll(t, sliceSize, numInputs, numRecovery, 1)
	for round := 0; round < 5; round++ {
		got := encodeAll(t, sliceSize, numInputs, numRecovery, 8)
		for i := range want {
			if !bytes.Equal(want[i], got[i]) {
				t.Fatalf("round %d: recovery block %d differs between sequential and concurrent Add", round, i)
			}
		}
	}
}
