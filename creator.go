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

	"github.com/javi11/par2go/internal/packets"
	"github.com/javi11/par2go/internal/parpar"
)

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
}

func (o *Options) withDefaults() Options {
	opts := *o
	if opts.NumGoroutines <= 0 {
		opts.NumGoroutines = runtime.NumCPU()
	}
	if opts.Creator == "" {
		opts.Creator = "Postie"
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

	// Phase 1: Hash input files and compute per-slice checksums
	slog.Debug("par2go: hashing input files", "count", len(inputFiles))
	report("hashing", 0)

	files, err := hashFiles(ctx, inputFiles, opts.SliceSize, func(pct float64) {
		report("hashing", pct)
	})
	if err != nil {
		return fmt.Errorf("par2go: hashing failed: %w", err)
	}

	report("hashing", 1.0)

	// Phase 2: Build Main packet and derive Recovery Set ID
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

	// Phase 3: RS encode recovery blocks via PAR2ProcCPU
	slog.Debug("par2go: encoding recovery blocks",
		"numRecovery", opts.NumRecovery,
		"sliceSize", opts.SliceSize)

	// Build a flat list of all input slices across files
	type sliceRef struct {
		fileIdx  int
		sliceIdx int
	}
	var allSlices []sliceRef
	for fi, f := range files {
		numSlices := int((f.size + uint64(opts.SliceSize) - 1) / uint64(opts.SliceSize))
		for s := range numSlices {
			allSlices = append(allSlices, sliceRef{fileIdx: fi, sliceIdx: s})
		}
	}

	totalInputSlices := len(allSlices)
	if totalInputSlices == 0 {
		return fmt.Errorf("par2go: no input slices (all files empty?)")
	}

	proc, err := parpar.NewGfProc(opts.SliceSize, opts.NumGoroutines)
	if err != nil {
		return fmt.Errorf("par2go: init encoder: %w", err)
	}
	defer proc.Close()

	slog.Debug("par2go: encoder ready", "method", proc.MethodName(), "threads", proc.NumThreads())

	exponents := make([]uint16, opts.NumRecovery)
	for i := range exponents {
		exponents[i] = uint16(i)
	}
	proc.SetRecoverySlices(exponents)

	// Open all input files upfront to avoid repeated open/close per slice
	openFiles := make([]*os.File, len(files))
	for i, fi := range files {
		f, err := os.Open(fi.path)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = openFiles[j].Close()
			}
			return fmt.Errorf("par2go: open %s: %w", fi.path, err)
		}
		openFiles[i] = f
	}
	defer func() {
		for _, f := range openFiles {
			_ = f.Close()
		}
	}()

	// Buffer pool to reuse slice-sized buffers across readSlice calls
	bufPool := sync.Pool{
		New: func() any {
			b := make([]byte, opts.SliceSize)
			return &b
		},
	}

	// Feed all input slices to the encoder
	for i, ref := range allSlices {
		if err := ctx.Err(); err != nil {
			return err
		}

		ptr := bufPool.Get().(*[]byte)
		buf := *ptr
		n, readErr := openFiles[ref.fileIdx].ReadAt(buf, int64(ref.sliceIdx)*int64(opts.SliceSize))
		if readErr != nil && readErr != io.EOF {
			bufPool.Put(ptr)
			return fmt.Errorf("par2go: read slice %d: %w", i, readErr)
		}
		if n < len(buf) {
			clear(buf[n:])
		}

		proc.Add(i, buf)
		bufPool.Put(ptr)

		report("encoding", float64(i+1)/float64(totalInputSlices))
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	proc.End()

	// Collect recovery blocks
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

	// Phase 4: Write output files
	slog.Debug("par2go: writing PAR2 files", "output", outputPath)
	report("writing", 0)

	// 4a: Write main .par2 file (no recovery data, just metadata packets)
	if err := writeMainFile(outputPath, recoverySetID, mainBody, files, opts.Creator); err != nil {
		return fmt.Errorf("par2go: writing main file failed: %w", err)
	}

	// 4b: Write volume files with doubling strategy
	if err := writeVolumeFiles(outputPath, recoverySetID, recoveryBlocks, opts.SliceSize); err != nil {
		return fmt.Errorf("par2go: writing volume files failed: %w", err)
	}

	report("writing", 1.0)
	slog.Debug("par2go: done")

	return nil
}

// hashFiles computes hashes and per-slice checksums for all input files.
func hashFiles(ctx context.Context, paths []string, sliceSize int, onProgress func(float64)) ([]fileInfo, error) {
	var totalSize int64
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return nil, fmt.Errorf("stat %s: %w", p, err)
		}
		totalSize += info.Size()
	}

	var processedSize int64
	files := make([]fileInfo, len(paths))

	for i, p := range paths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		fi, err := hashSingleFile(ctx, p, sliceSize, func(bytesRead int64) {
			if totalSize > 0 {
				onProgress(float64(processedSize+bytesRead) / float64(totalSize))
			}
		})
		if err != nil {
			return nil, err
		}
		files[i] = fi

		processedSize += int64(fi.size)
		if totalSize > 0 {
			onProgress(float64(processedSize) / float64(totalSize))
		}
	}

	return files, nil
}

// hashSingleFile computes all hashes and checksums for a single file.
// onProgress is called after each slice with the total bytes read so far.
func hashSingleFile(ctx context.Context, path string, sliceSize int, onProgress func(bytesRead int64)) (fileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return fileInfo{}, fmt.Errorf("open %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	stat, err := f.Stat()
	if err != nil {
		return fileInfo{}, err
	}

	fi := fileInfo{
		path: path,
		name: filepath.Base(path),
		size: uint64(stat.Size()),
	}

	// Compute all hashes in a single pass
	hashFull := md5.New()
	hash16k := md5.New()

	numSlices := int((fi.size + uint64(sliceSize) - 1) / uint64(sliceSize))
	if fi.size == 0 {
		numSlices = 0
	}
	fi.slices = make([]packets.IFSCEntry, numSlices)

	buf := make([]byte, sliceSize)
	var totalRead int64
	sliceIdx := 0

	for {
		if err := ctx.Err(); err != nil {
			return fileInfo{}, err
		}

		n, err := io.ReadFull(f, buf)
		if n == 0 {
			if err == io.EOF {
				break
			}
			if err != nil {
				return fileInfo{}, fmt.Errorf("read %s: %w", path, err)
			}
		}

		slice := buf[:n]

		// Full file hash
		hashFull.Write(slice)

		// 16K hash
		if totalRead < 16384 {
			end := int64(n)
			if totalRead+end > 16384 {
				end = 16384 - totalRead
			}
			hash16k.Write(slice[:end])
		}

		// Per-slice checksums (pad last slice with zeros)
		if n < sliceSize {
			padded := make([]byte, sliceSize)
			copy(padded, slice)
			slice = padded
		}
		fi.slices[sliceIdx] = packets.IFSCEntry{
			MD5:   md5.Sum(slice),
			CRC32: crc32Sum(slice),
		}
		sliceIdx++

		totalRead += int64(n)
		onProgress(totalRead)

		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
	}

	copy(fi.hashFull[:], hashFull.Sum(nil))
	copy(fi.hash16k[:], hash16k.Sum(nil))
	fi.fileID = packets.FileID(fi.hash16k, fi.size, fi.name)

	return fi, nil
}

// writeMainFile writes the main .par2 file containing metadata packets.
func writeMainFile(outputPath string, recoverySetID [16]byte, mainBody []byte, files []fileInfo, creator string) error {
	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	// Write Main packet
	if err := packets.WriteHeader(f, recoverySetID, packets.TypeMain, mainBody); err != nil {
		return err
	}

	// Write Creator packet
	creatorBody := packets.CreatorPacket(creator)
	if err := packets.WriteHeader(f, recoverySetID, packets.TypeCreator, creatorBody); err != nil {
		return err
	}

	// Write File Description + IFSC packets for each file
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

// writeVolumeFiles writes .volN+M.par2 files using a doubling strategy.
// Block counts per volume: 1, 1, 2, 4, 8, 16, ... until all blocks are placed.
func writeVolumeFiles(outputPath string, recoverySetID [16]byte, blocks []recoveryBlock, sliceSize int) error {
	if len(blocks) == 0 {
		return nil
	}

	// Sort blocks by exponent
	sort.Slice(blocks, func(i, j int) bool {
		return blocks[i].exponent < blocks[j].exponent
	})

	// Strip .par2 extension for volume file naming
	base := strings.TrimSuffix(outputPath, filepath.Ext(outputPath))

	offset := 0
	count := 1 // Start with 1, then 1, 2, 4, 8, ...
	firstVolume := true

	for offset < len(blocks) {
		end := offset + count
		if end > len(blocks) {
			end = len(blocks)
		}

		actualCount := end - offset
		volName := fmt.Sprintf("%s.vol%02d+%02d.par2", base, offset, actualCount)

		if err := writeVolumeFile(volName, recoverySetID, blocks[offset:end]); err != nil {
			return err
		}

		offset = end

		// Doubling: first two volumes have 1 block each, then 2, 4, 8, ...
		if firstVolume {
			firstVolume = false
			// count stays 1 for second volume
		} else {
			count *= 2
		}
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
