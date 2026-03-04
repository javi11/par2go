// Package par2go implements a pure Go PAR2 parity file creator.
//
// It creates PAR2 recovery files compatible with par2cmdline, MultiPar, and
// other PAR2-compliant tools, without requiring any external binaries.
package par2go

import (
	"bytes"
	"context"
	"crypto/md5"
	"fmt"
	"hash/crc32"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/javi11/par2go/internal/packets"
	"github.com/javi11/par2go/internal/parpar"
)

// discardHandler is a no-op slog handler used as the default logger for the library.
var discardHandler = slog.New(slog.NewTextHandler(io.Discard, nil))

// Options configures PAR2 creation.
type Options struct {
	// SliceSize is the block/slice size in bytes. Must be a multiple of 4.
	SliceSize int
	// NumRecovery is the number of recovery blocks to create.
	NumRecovery int
	// NumGoroutines is the number of parallel GF compute threads (default: runtime.NumCPU()).
	// Pass 0 to use hardware_concurrency() auto-detection.
	NumGoroutines int
	// OnProgress reports progress: phase is "hashing", "encoding", or "writing", pct is 0.0-1.0.
	OnProgress func(phase string, pct float64)
	// Creator is the creator string embedded in the PAR2 file (default: "Postie").
	Creator string
	// Logger is the structured logger used internally. Defaults to discarding all output.
	// Set to slog.Default() or a custom *slog.Logger to enable logging.
	Logger *slog.Logger
}

func (o *Options) withDefaults() Options {
	opts := *o
	if opts.NumGoroutines <= 0 {
		opts.NumGoroutines = runtime.NumCPU()
	}
	if opts.Creator == "" {
		opts.Creator = "Postie"
	}
	if opts.Logger == nil {
		opts.Logger = discardHandler
	}
	return opts
}

// recoveryBlock holds a single recovery block's exponent and data.
type recoveryBlock struct {
	exponent uint16
	data     []byte
}

// fileInfo holds computed metadata for a single input file.
type fileInfo struct {
	path     string
	name     string // basename only, used in PAR2 packets
	size     uint64
	hash16k  [16]byte // MD5 of first 16KB
	hashFull [16]byte // MD5 of entire file
	fileID   [16]byte
	slices   []packets.IFSCEntry
}

// fileNumSlices returns the number of PAR2 slices for a file of the given size.
func fileNumSlices(size uint64, sliceSize int) int {
	if size == 0 {
		return 0
	}
	return int((size + uint64(sliceSize) - 1) / uint64(sliceSize))
}

// Create creates PAR2 parity files for the given input files.
//
// outputPath is the path for the main .par2 file (e.g., "/path/to/file.par2").
// Volume files will be created alongside with names like file.vol00+01.par2.
func Create(ctx context.Context, outputPath string, inputFiles []string, opts Options) error {
	opts = opts.withDefaults()

	if opts.SliceSize <= 0 || opts.SliceSize%4 != 0 {
		return fmt.Errorf("par2go: SliceSize must be a positive multiple of 4, got %d", opts.SliceSize)
	}
	if opts.NumRecovery <= 0 {
		return fmt.Errorf("par2go: NumRecovery must be positive, got %d", opts.NumRecovery)
	}
	if len(inputFiles) == 0 {
		return fmt.Errorf("par2go: no input files")
	}

	report := func(phase string, pct float64) {
		if opts.OnProgress != nil {
			opts.OnProgress(phase, pct)
		}
	}

	// Phase 1: Quick scan — stat + read first 16KB per file in parallel.
	// Provides hash16k, size, and fileID needed for the Main packet.
	opts.Logger.Debug("par2go: scanning input files", "count", len(inputFiles))
	report("hashing", 0)

	files, err := quickScanFiles(ctx, inputFiles)
	if err != nil {
		return fmt.Errorf("par2go: scanning failed: %w", err)
	}

	// Phase 2: Build Main packet and derive Recovery Set ID.
	fileIDs := make([][16]byte, len(files))
	for i, f := range files {
		fileIDs[i] = f.fileID
	}
	// PAR2 spec requires file IDs sorted by value (unsigned 128-bit integer).
	slices.SortFunc(fileIDs, func(a, b [16]byte) int {
		return bytes.Compare(a[:], b[:])
	})
	mainBody := packets.MainPacket(uint64(opts.SliceSize), fileIDs)
	recoverySetID := packets.RecoverySetID(mainBody)

	// Phase 3: Single-pass hash+encode.
	// Each file is read exactly once: hashFull and IFSC are computed while
	// simultaneously feeding the RS encoder.  IFSC computation is offloaded
	// to a pool of goroutines to overlap with I/O and GF16 compute.
	opts.Logger.Debug("par2go: encoding recovery blocks",
		"numRecovery", opts.NumRecovery,
		"sliceSize", opts.SliceSize)

	proc, err := parpar.NewGfProc(opts.SliceSize, opts.NumGoroutines)
	if err != nil {
		return fmt.Errorf("par2go: init encoder: %w", err)
	}
	defer proc.Close()

	opts.Logger.Debug("par2go: encoder ready", "method", proc.MethodName(), "threads", proc.NumThreads())

	exponents := make([]uint16, opts.NumRecovery)
	for i := range exponents {
		exponents[i] = uint16(i)
	}
	proc.SetRecoverySlices(exponents)

	if err := hashAndEncodeFiles(ctx, files, proc, opts.SliceSize, func(pct float64) {
		report("hashing", pct)
	}); err != nil {
		return fmt.Errorf("par2go: hash+encode failed: %w", err)
	}

	report("hashing", 1.0)

	if err := ctx.Err(); err != nil {
		return err
	}

	proc.End()

	// Collect recovery blocks.
	recoveryBlocks := make([]recoveryBlock, opts.NumRecovery)
	for i := range recoveryBlocks {
		recoveryBlocks[i] = recoveryBlock{
			exponent: uint16(i),
			data:     make([]byte, opts.SliceSize),
		}
		proc.GetOutput(i, recoveryBlocks[i].data)
	}
	proc.FreeMem()

	report("encoding", 1.0)

	// Phase 4: Write output files (volume files written in parallel).
	opts.Logger.Debug("par2go: writing PAR2 files", "output", outputPath)
	report("writing", 0)

	if err := writeMainFile(outputPath, recoverySetID, mainBody, files, opts.Creator); err != nil {
		return fmt.Errorf("par2go: writing main file failed: %w", err)
	}
	if err := writeVolumeFiles(outputPath, recoverySetID, recoveryBlocks, opts.SliceSize); err != nil {
		return fmt.Errorf("par2go: writing volume files failed: %w", err)
	}

	report("writing", 1.0)
	opts.Logger.Debug("par2go: done")
	return nil
}

// quickScanFile reads file metadata and the first 16KB to compute hash16k and fileID.
func quickScanFile(path string) (fileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return fileInfo{}, fmt.Errorf("open %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	stat, err := f.Stat()
	if err != nil {
		return fileInfo{}, fmt.Errorf("stat %s: %w", path, err)
	}

	fi := fileInfo{
		path: path,
		name: filepath.Base(path),
		size: uint64(stat.Size()),
	}

	if stat.Size() > 0 {
		readSize := stat.Size()
		if readSize > 16384 {
			readSize = 16384
		}
		buf := make([]byte, readSize)
		if _, err := io.ReadFull(f, buf); err != nil {
			return fileInfo{}, fmt.Errorf("read16k %s: %w", path, err)
		}
		fi.hash16k = md5.Sum(buf)
	}

	fi.fileID = packets.FileID(fi.hash16k, fi.size, fi.name)
	return fi, nil
}

// quickScanFiles runs quickScanFile on all paths concurrently.
func quickScanFiles(ctx context.Context, paths []string) ([]fileInfo, error) {
	files := make([]fileInfo, len(paths))
	errc := make(chan error, len(paths))

	var wg sync.WaitGroup
	for i, p := range paths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		wg.Add(1)
		go func(i int, p string) {
			defer wg.Done()
			fi, err := quickScanFile(p)
			if err != nil {
				errc <- err
				return
			}
			files[i] = fi
		}(i, p)
	}
	wg.Wait()
	close(errc)
	for err := range errc {
		return nil, err
	}
	return files, nil
}

// hashAndEncodeFiles performs a single pass over all input files:
//   - Reader goroutines (one per file, bounded by numCPU) read slices from disk,
//     update hashFull incrementally, and call proc.Add inline.
//   - A pool of IFSC hash goroutines computes per-slice MD5+CRC32 concurrently,
//     overlapping with I/O and GF16 compute.
//
// This eliminates the double-read and decouples slow per-slice hashing from I/O.
func hashAndEncodeFiles(
	ctx context.Context,
	files []fileInfo,
	proc *parpar.GfProc,
	sliceSize int,
	onProgress func(float64),
) error {
	// Pre-assign global slice offsets per file.
	offsets := make([]int, len(files))
	totalSlices := 0
	for i := range files {
		offsets[i] = totalSlices
		totalSlices += fileNumSlices(files[i].size, sliceSize)
	}
	if totalSlices == 0 {
		return fmt.Errorf("par2go: no input slices (all files empty?)")
	}

	// Pre-allocate IFSC slice arrays so hash workers can write directly.
	for i := range files {
		n := fileNumSlices(files[i].size, sliceSize)
		if n > 0 {
			files[i].slices = make([]packets.IFSCEntry, n)
		}
	}

	// Open all files upfront.
	openFiles := make([]*os.File, len(files))
	for i := range files {
		f, err := os.Open(files[i].path)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = openFiles[j].Close()
			}
			return fmt.Errorf("par2go: open %s: %w", files[i].path, err)
		}
		openFiles[i] = f
	}
	defer func() {
		for _, f := range openFiles {
			_ = f.Close()
		}
	}()

	// Buffer pool: reuses slice-sized buffers across all goroutines.
	bufPool := &sync.Pool{
		New: func() any {
			b := make([]byte, sliceSize)
			return &b
		},
	}

	// IFSC hash pool: per-slice MD5+CRC32 runs concurrently with I/O and GF16.
	// Decoupling this from the reader goroutine lets I/O proceed at full speed.
	numHashWorkers := runtime.NumCPU() / 2
	if numHashWorkers < 1 {
		numHashWorkers = 1
	}
	if numHashWorkers > 8 {
		numHashWorkers = 8
	}

	type hashWork struct {
		bptr     *[]byte
		fileIdx  int
		sliceIdx int
	}

	hashCh := make(chan hashWork, numHashWorkers*4)

	var hashWg sync.WaitGroup
	for j := 0; j < numHashWorkers; j++ {
		hashWg.Add(1)
		go func() {
			defer hashWg.Done()
			for hw := range hashCh {
				buf := *hw.bptr
				files[hw.fileIdx].slices[hw.sliceIdx] = packets.IFSCEntry{
					MD5:   md5.Sum(buf),
					CRC32: crc32Sum(buf),
				}
				bufPool.Put(hw.bptr)
			}
		}()
	}

	// Reader goroutines: read slices, update hashFull, call proc.Add inline.
	// proc.Add is safe for concurrent calls from multiple goroutines.
	numReaders := len(files)
	if numReaders > runtime.NumCPU() {
		numReaders = runtime.NumCPU()
	}
	if numReaders < 1 {
		numReaders = 1
	}
	sem := make(chan struct{}, numReaders)

	var readerErrs []error
	var readerMu sync.Mutex
	var readerWg sync.WaitGroup
	var doneSlices atomic.Int64

	for i := range files {
		readerWg.Add(1)
		go func(i int) {
			defer readerWg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fi := &files[i]
			f := openFiles[i]
			hashFull := md5.New()
			sliceIdx := 0

			for {
				if err := ctx.Err(); err != nil {
					readerMu.Lock()
					readerErrs = append(readerErrs, err)
					readerMu.Unlock()
					return
				}

				bptr := bufPool.Get().(*[]byte)
				buf := *bptr

				n, err := io.ReadFull(f, buf)
				if n == 0 {
					bufPool.Put(bptr)
					if err == io.EOF {
						break
					}
					readerMu.Lock()
					readerErrs = append(readerErrs, fmt.Errorf("read %s: %w", fi.path, err))
					readerMu.Unlock()
					return
				}

				// Hash only actual file bytes (not zero-padding).
				hashFull.Write(buf[:n])

				// Zero-pad last partial slice for encoding and IFSC.
				if n < sliceSize {
					clear(buf[n:])
				}

				// Feed encoder inline — no channel round-trip needed.
				// proc.Add is concurrency-safe across goroutines.
				proc.Add(offsets[i]+sliceIdx, buf)

				// Delegate per-slice IFSC hashing to the hash pool.
				// The buffer is still valid; hash workers own it until they put it back.
				hw := hashWork{bptr: bptr, fileIdx: i, sliceIdx: sliceIdx}
				select {
				case hashCh <- hw:
				case <-ctx.Done():
					bufPool.Put(bptr)
					readerMu.Lock()
					readerErrs = append(readerErrs, ctx.Err())
					readerMu.Unlock()
					return
				}

				sliceIdx++
				done := doneSlices.Add(1)
				onProgress(float64(done) / float64(totalSlices))

				if err == io.EOF || err == io.ErrUnexpectedEOF {
					break
				}
			}

			copy(fi.hashFull[:], hashFull.Sum(nil))
		}(i)
	}

	// Close hash channel once all readers are done so hash workers can exit.
	readerWg.Wait()
	close(hashCh)
	hashWg.Wait()

	if err := ctx.Err(); err != nil {
		return err
	}

	readerMu.Lock()
	defer readerMu.Unlock()
	if len(readerErrs) > 0 {
		return readerErrs[0]
	}
	return nil
}

// writeMainFile writes the main .par2 file containing metadata packets.
func writeMainFile(outputPath string, recoverySetID [16]byte, mainBody []byte, files []fileInfo, creator string) error {
	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	if err := packets.WriteHeader(f, recoverySetID, packets.TypeMain, mainBody); err != nil {
		return err
	}

	creatorBody := packets.CreatorPacket(creator)
	if err := packets.WriteHeader(f, recoverySetID, packets.TypeCreator, creatorBody); err != nil {
		return err
	}

	for _, fi := range files {
		fdBody := packets.FileDescriptionPacket(fi.fileID, fi.hashFull, fi.hash16k, fi.size, fi.name)
		if err := packets.WriteHeader(f, recoverySetID, packets.TypeFileDescription, fdBody); err != nil {
			return err
		}

		ifscBody := packets.IFSCPacket(fi.fileID, fi.slices)
		if err := packets.WriteHeader(f, recoverySetID, packets.TypeIFSC, ifscBody); err != nil {
			return err
		}
	}

	return f.Close()
}

// writeVolumeFiles writes .volN+M.par2 files using a doubling strategy,
// with volume files written in parallel (bounded by numCPU).
//
// Block counts per volume: 1, 1, 2, 4, 8, 16, ... until all blocks are placed.
func writeVolumeFiles(outputPath string, recoverySetID [16]byte, blocks []recoveryBlock, sliceSize int) error {
	if len(blocks) == 0 {
		return nil
	}

	sort.Slice(blocks, func(i, j int) bool {
		return blocks[i].exponent < blocks[j].exponent
	})

	base := strings.TrimSuffix(outputPath, filepath.Ext(outputPath))

	type volGroup struct {
		name   string
		blocks []recoveryBlock
	}
	var groups []volGroup

	offset := 0
	count := 1
	firstVolume := true

	for offset < len(blocks) {
		end := offset + count
		if end > len(blocks) {
			end = len(blocks)
		}
		actualCount := end - offset
		volName := fmt.Sprintf("%s.vol%02d+%02d.par2", base, offset, actualCount)
		groups = append(groups, volGroup{name: volName, blocks: blocks[offset:end]})
		offset = end

		if firstVolume {
			firstVolume = false
		} else {
			count *= 2
		}
	}

	// Write all volume files concurrently.
	numWorkers := len(groups)
	if numWorkers > runtime.NumCPU() {
		numWorkers = runtime.NumCPU()
	}
	sem := make(chan struct{}, numWorkers)
	errc := make(chan error, len(groups))

	var wg sync.WaitGroup
	for _, g := range groups {
		wg.Add(1)
		go func(g volGroup) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if err := writeVolumeFile(g.name, recoverySetID, g.blocks); err != nil {
				errc <- err
			}
		}(g)
	}
	wg.Wait()
	close(errc)

	for err := range errc {
		return err
	}
	return nil
}

// writeVolumeFile writes a single volume file.
func writeVolumeFile(path string, recoverySetID [16]byte, blocks []recoveryBlock) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}

	for _, block := range blocks {
		body := packets.RecoverySlicePacket(block.exponent, block.data)
		if err := packets.WriteHeader(f, recoverySetID, packets.TypeRecoverySlice, body); err != nil {
			_ = f.Close()
			return err
		}
	}

	return f.Close()
}

// crc32Sum computes CRC32/IEEE of data using the hardware-accelerated standard library.
func crc32Sum(data []byte) uint32 {
	return crc32.ChecksumIEEE(data)
}
