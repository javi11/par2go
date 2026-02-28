// Package rsenc implements PAR2-compatible Reed-Solomon encoding over GF(2^16).
//
// PAR2 uses a specific set of constants (powers of 2 in GF(2^16), filtered by
// divisibility constraints) and a Vandermonde-like encoding matrix. This package
// implements batched encoding with configurable memory budgets for large files.
package rsenc

import (
	"context"
	"runtime"

	"github.com/javi11/par2go/internal/gf16"
)

// DefaultMemoryBudget is the default memory budget for recovery block buffers (512 MB).
const DefaultMemoryBudget = 512 * 1024 * 1024

// Encoder performs PAR2-compatible Reed-Solomon encoding.
type Encoder struct {
	sliceSize    int
	numRecovery  int
	memoryBudget int
	numWorkers   int
	exponents    []uint16 // recovery block exponents
}

// NewEncoder creates a new RS encoder for the given parameters.
func NewEncoder(sliceSize, numRecovery int) *Encoder {
	return &Encoder{
		sliceSize:    sliceSize,
		numRecovery:  numRecovery,
		memoryBudget: DefaultMemoryBudget,
		numWorkers:   runtime.NumCPU(),
		exponents:    generateExponents(numRecovery),
	}
}

// SetMemoryBudget sets the maximum memory to use for recovery block buffers.
func (e *Encoder) SetMemoryBudget(bytes int) {
	if bytes > 0 {
		e.memoryBudget = bytes
	}
}

// SetNumWorkers sets the number of concurrent goroutines for encoding.
func (e *Encoder) SetNumWorkers(n int) {
	if n > 0 {
		e.numWorkers = n
	}
}

// Exponents returns the exponents used for recovery blocks.
// These correspond to the recovery block numbering in PAR2 packets.
func (e *Encoder) Exponents() []uint16 {
	return e.exponents
}

// generateExponents creates recovery block exponents starting from 0.
// In PAR2, recovery block e has exponent e (0-indexed).
func generateExponents(n int) []uint16 {
	exps := make([]uint16, n)
	for i := range exps {
		exps[i] = uint16(i)
	}
	return exps
}

// GenerateConstants generates the PAR2 encoding constants for the given number of input slices.
//
// Per the PAR2 spec, constants are successive powers of 2 in GF(2^16), skipping values
// where the exponent n satisfies: n%3==0 || n%5==0 || n%17==0 || n%257==0.
// This ensures the Vandermonde matrix has the required properties.
func GenerateConstants(numInputSlices int) []uint16 {
	constants := make([]uint16, 0, numInputSlices)
	n := 0
	for len(constants) < numInputSlices {
		if n%3 != 0 && n%5 != 0 && n%17 != 0 && n%257 != 0 {
			constants = append(constants, gf16.Pow(2, uint16(n)))
		}
		n++
		// Safety: GF(2^16) has 65535 non-zero elements
		if n > 65535 {
			break
		}
	}
	return constants
}

// Process encodes recovery blocks from input slices.
//
// It reads input slices via readSlice, computes recovery data, and writes
// completed recovery blocks via writeRecovery. Processing is batched to
// stay within the configured memory budget.
//
// If releaseSlice is non-nil, it is called after each slice is consumed
// to allow the caller to return buffers to a pool.
//
// For each recovery block e and input slice i:
//
//	recoveryBlock[e] ^= inputSlice[i] * constant[i] ^ exponent[e]
//
// The onProgress callback (if non-nil) receives values from 0.0 to 1.0.
func (e *Encoder) Process(
	ctx context.Context,
	numInputSlices int,
	readSlice func(i int) ([]byte, error),
	releaseSlice func([]byte),
	writeRecovery func(exponent uint16, data []byte) error,
	onProgress func(float64),
) error {
	if numInputSlices == 0 || e.numRecovery == 0 {
		return nil
	}

	constants := GenerateConstants(numInputSlices)

	// Calculate batch size based on memory budget
	batchSize := e.memoryBudget / e.sliceSize
	if batchSize > e.numRecovery {
		batchSize = e.numRecovery
	}
	if batchSize <= 0 {
		batchSize = 1
	}

	totalWork := float64(numInputSlices) * float64(e.numRecovery)
	doneWork := float64(0)

	// Persistent worker pool: created once and reused across all batches and
	// slices, eliminating goroutine churn from the old parallelAccumulate.
	type accTask struct {
		start, end int
		constant   uint16
		batchStart int
	}
	numWorkers := e.numWorkers

	// Shared state updated by the coordinator before dispatching tasks.
	// Channel operations (send to taskCh / receive from doneCh) provide
	// happens-before ordering so workers always see the latest values.
	var activeRecoveryBlocks [][]byte
	var activeSliceData []byte

	// Serial accumulator used for the non-parallel path and for FinishBlock.
	serialAcc := gf16.NewBatchAccumulator(e.sliceSize)
	defer serialAcc.Free()

	var taskCh chan accTask
	var doneCh chan struct{}

	if numWorkers > 1 {
		taskCh = make(chan accTask, numWorkers)
		doneCh = make(chan struct{}, numWorkers)

		for w := 0; w < numWorkers; w++ {
			go func() {
				ba := gf16.NewBatchAccumulator(e.sliceSize)
				defer ba.Free()
				for task := range taskCh {
					ba.PrepareInput(activeSliceData)
					for j := task.start; j < task.end; j++ {
						exp := e.exponents[task.batchStart+j]
						factor := gf16.Pow(task.constant, exp)
						ba.AccumulatePrepared(activeRecoveryBlocks[j], factor)
					}
					doneCh <- struct{}{}
				}
			}()
		}
		defer close(taskCh)
	}

	// Double-buffered I/O: pre-read the next input slice while the current
	// one is being processed, overlapping disk I/O with compute.
	type prefetchResult struct {
		data []byte
		err  error
	}
	prefetchCh := make(chan prefetchResult, 1)

	// Process recovery blocks in batches
	for batchStart := 0; batchStart < e.numRecovery; batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > e.numRecovery {
			batchEnd = e.numRecovery
		}
		batchCount := batchEnd - batchStart

		// Allocate aligned recovery block buffers for this batch.
		// Aligned memory is required for MulAddPacked and Finish.
		recoveryBlocks := make([][]byte, batchCount)
		for j := range recoveryBlocks {
			recoveryBlocks[j] = gf16.AlignedSlice(e.sliceSize)
		}

		// Kick off first read for this batch
		go func() {
			d, err := readSlice(0)
			prefetchCh <- prefetchResult{d, err}
		}()

		// Read each input slice and accumulate into all recovery blocks in this batch
		for i := 0; i < numInputSlices; i++ {
			if err := ctx.Err(); err != nil {
				// Drain pending prefetch to avoid goroutine leak
				result := <-prefetchCh
				if releaseSlice != nil && result.data != nil {
					releaseSlice(result.data)
				}
				for k := range recoveryBlocks {
					gf16.FreeAligned(recoveryBlocks[k])
				}
				return err
			}

			// Receive prefetched slice
			result := <-prefetchCh
			if result.err != nil {
				for k := range recoveryBlocks {
					gf16.FreeAligned(recoveryBlocks[k])
				}
				return result.err
			}
			sliceData := result.data

			// Pre-read next slice while processing current one
			if i+1 < numInputSlices {
				nextIdx := i + 1
				go func() {
					d, err := readSlice(nextIdx)
					prefetchCh <- prefetchResult{d, err}
				}()
			}

			// Dispatch to worker pool or process inline
			if taskCh != nil && batchCount > numWorkers {
				activeRecoveryBlocks = recoveryBlocks
				activeSliceData = sliceData

				blocksPerWorker := (batchCount + numWorkers - 1) / numWorkers
				tasksDispatched := 0
				for w := 0; w < numWorkers; w++ {
					start := w * blocksPerWorker
					end := start + blocksPerWorker
					if end > batchCount {
						end = batchCount
					}
					if start >= end {
						break
					}
					taskCh <- accTask{
						start:      start,
						end:        end,
						constant:   constants[i],
						batchStart: batchStart,
					}
					tasksDispatched++
				}
				for range tasksDispatched {
					<-doneCh
				}
			} else {
				// Serial: not enough blocks to warrant parallelism
				serialAcc.PrepareInput(sliceData)
				for j := 0; j < batchCount; j++ {
					exp := e.exponents[batchStart+j]
					factor := gf16.Pow(constants[i], exp)
					serialAcc.AccumulatePrepared(recoveryBlocks[j], factor)
				}
			}

			if releaseSlice != nil {
				releaseSlice(sliceData)
			}

			doneWork += float64(batchCount)
			if onProgress != nil {
				onProgress(doneWork / totalWork)
			}
		}

		// Convert recovery blocks from packed format back to raw, then
		// copy each to a Go-managed slice and free the aligned buffer.
		// This keeps the writeRecovery callback's memory contract unchanged.
		for j := 0; j < batchCount; j++ {
			serialAcc.FinishBlock(recoveryBlocks[j])
		}

		// Write completed recovery blocks
		for j := 0; j < batchCount; j++ {
			if err := ctx.Err(); err != nil {
				// Free remaining aligned blocks on early exit
				for k := j; k < batchCount; k++ {
					gf16.FreeAligned(recoveryBlocks[k])
				}
				return err
			}
			exponent := e.exponents[batchStart+j]
			// Copy to GC-managed memory and immediately free the aligned block.
			goBuf := make([]byte, e.sliceSize)
			copy(goBuf, recoveryBlocks[j])
			gf16.FreeAligned(recoveryBlocks[j])
			if err := writeRecovery(exponent, goBuf); err != nil {
				// Free remaining aligned blocks on callback error
				for k := j + 1; k < batchCount; k++ {
					gf16.FreeAligned(recoveryBlocks[k])
				}
				return err
			}
		}
	}

	return nil
}
