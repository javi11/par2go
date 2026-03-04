// Command benchmark compares par2go throughput against the parpar binary.
//
// Usage:
//
//	go run ./cmd/benchmark [-input FILE] [-parpar PATH]
//
// If the input file does not exist it is generated with pseudo-random data.
package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	par2go "github.com/javi11/par2go"
)

const (
	benchFileSizeBytes = 1024 * 1024 * 1024 // 1 GiB
	benchSliceSize     = 768 * 1024          // 768 KB (matches parpar default benchmark)
	benchNumRecovery   = 10
)

func main() {
	inputFlag := flag.String("input", "/tmp/par2go_bench.bin", "input file (created with random data if absent)")
	parparFlag := flag.String("parpar", "", "path to parpar binary")
	flag.Parse()

	if err := ensureTestFile(*inputFlag, benchFileSizeBytes); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	par2goMBs := runPar2go(*inputFlag)
	fmt.Println()
	parparMBs := runParpar(*parparFlag, *inputFlag)

	fmt.Println()
	fmt.Println("=== Results ===")
	fmt.Printf("par2go : %7.1f MB/s\n", par2goMBs)
	fmt.Printf("parpar : %7.1f MB/s\n", parparMBs)
	if parparMBs > 0 {
		fmt.Printf("ratio  : %7.1f%%\n", par2goMBs/parparMBs*100)
	}
}

// ensureTestFile creates path with size bytes of pseudo-random content if it
// does not already exist at the right size.
func ensureTestFile(path string, size int64) error {
	if info, err := os.Stat(path); err == nil && info.Size() == size {
		fmt.Printf("Reusing existing test file: %s (%.2f GiB)\n", path, float64(size)/float64(1<<30))
		return nil
	}

	fmt.Printf("Generating %s (%.2f GiB)...\n", path, float64(size)/float64(1<<30))

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	buf := make([]byte, 4*1024*1024)
	rng := rand.New(rand.NewPCG(0xdeadbeef, 0xcafebabe))
	var written int64
	for written < size {
		for i := range buf {
			buf[i] = byte(rng.Uint32())
		}
		n := int64(len(buf))
		if written+n > size {
			n = size - written
		}
		if _, err := f.Write(buf[:n]); err != nil {
			return err
		}
		written += n
	}
	fmt.Printf("Generated: %s\n", path)
	return nil
}

// runPar2go benchmarks par2go and returns MB/s throughput.
func runPar2go(inputPath string) float64 {
	const outputPath = "/tmp/bench_par2go.par2"
	cleanupPar2(outputPath)
	defer cleanupPar2(outputPath)

	info, err := os.Stat(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "stat %s: %v\n", inputPath, err)
		return 0
	}
	fileMB := float64(info.Size()) / (1024 * 1024)

	opts := par2go.Options{
		SliceSize:   benchSliceSize,
		NumRecovery: benchNumRecovery,
	}

	fmt.Printf("par2go: starting (%.0f MB, slice=%dKB, recovery=%d)...\n",
		fileMB, benchSliceSize/1024, benchNumRecovery)

	start := time.Now()
	if err := par2go.Create(context.Background(), outputPath, []string{inputPath}, opts); err != nil {
		fmt.Fprintf(os.Stderr, "par2go error: %v\n", err)
		return 0
	}
	elapsed := time.Since(start)

	mbs := fileMB / elapsed.Seconds()
	fmt.Printf("par2go: %.1f MB/s  (%.2fs for %.0f MB)\n", mbs, elapsed.Seconds(), fileMB)
	return mbs
}

// runParpar benchmarks the parpar binary and returns MB/s throughput.
func runParpar(binPath, inputPath string) float64 {
	if _, err := os.Stat(binPath); err != nil {
		fmt.Printf("parpar: binary not found at %s (skipping)\n", binPath)
		return 0
	}

	const outputPath = "/tmp/bench_parpar.par2"
	cleanupPar2(outputPath)
	defer cleanupPar2(outputPath)

	info, err := os.Stat(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "stat %s: %v\n", inputPath, err)
		return 0
	}
	fileMB := float64(info.Size()) / (1024 * 1024)

	args := []string{
		"-s", fmt.Sprintf("%dk", benchSliceSize/1024),
		"-r", fmt.Sprintf("%d", benchNumRecovery),
		"-o", outputPath,
		"--", inputPath,
	}

	fmt.Printf("parpar: starting (%.0f MB, slice=%dKB, recovery=%d)...\n",
		fileMB, benchSliceSize/1024, benchNumRecovery)

	start := time.Now()
	cmd := exec.Command(binPath, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "parpar error: %v\n%s\n", err, string(out))
		return 0
	}
	elapsed := time.Since(start)

	mbs := fileMB / elapsed.Seconds()
	fmt.Printf("parpar: %.1f MB/s  (%.2fs for %.0f MB)\n", mbs, elapsed.Seconds(), fileMB)
	return mbs
}

// cleanupPar2 removes the given .par2 file and any .vol*.par2 siblings.
func cleanupPar2(path string) {
	dir := filepath.Dir(path)
	base := strings.TrimSuffix(filepath.Base(path), ".par2")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		n := e.Name()
		if strings.HasPrefix(n, base) && strings.HasSuffix(n, ".par2") {
			_ = os.Remove(filepath.Join(dir, n))
		}
	}
}
