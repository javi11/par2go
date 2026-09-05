package parpar

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

const kernelEnv = "PARPAR_KERNEL_UNDER_TEST"

// kernelSliceSizes mirrors sizes seen in the wild: postie's 128-aligned
// article-derived size, this package's unaligned default, and a tiny slice.
var kernelSliceSizes = []int{9984, 10000, 512}

// encodeWithMethod encodes deterministic slices with a forced GF16 method
// and returns the recovery blocks plus the kernel name ParPar actually used.
func encodeWithMethod(method, sliceSize, numInputs, numRecovery int, report bool) (string, [][]byte, error) {
	proc, err := NewGfProcWithConfig(GfProcConfig{SliceSize: sliceSize, NumThreads: 4, Method: method})
	if err != nil {
		return "", nil, err
	}
	defer proc.Close()
	name := proc.MethodName()
	if report {
		fmt.Printf("kernel=%q sliceSize=%d threads=%d stride=%d alignment=%d chunkLen=%d inputBatchSize=%d allocSliceSize=%d\n",
			name, sliceSize, proc.NumThreads(), proc.Stride(), proc.Alignment(), proc.ChunkLen(), proc.InputBatchSize(), proc.AllocSliceSize())
	}

	exps := make([]uint16, numRecovery)
	for i := range exps {
		exps[i] = uint16(i)
	}
	proc.SetRecoverySlices(exps)
	for i := 0; i < numInputs; i++ {
		data := make([]byte, sliceSize)
		for j := range data {
			data[j] = byte(i*13 + j*3)
		}
		proc.Add(i, data)
	}
	proc.End()

	out := make([][]byte, numRecovery)
	for i := range out {
		out[i] = make([]byte, sliceSize)
		proc.GetOutput(i, out[i])
	}
	return name, out, nil
}

// TestKernelSubprocess is the body run in a child process by
// TestEveryAvailableKernel. It exits non-zero on a mismatch and simply
// dies on a kernel crash, which the parent reports per method.
func TestKernelSubprocess(t *testing.T) {
	raw := os.Getenv(kernelEnv)
	if raw == "" {
		t.Skip("only runs as a subprocess of TestEveryAvailableKernel")
	}
	method, err := strconv.Atoi(raw)
	if err != nil {
		t.Fatal(err)
	}
	const numInputs, numRecovery = 40, 3
	for _, sliceSize := range kernelSliceSizes {
		_, want, err := encodeWithMethod(GF16Lookup, sliceSize, numInputs, numRecovery, false)
		if err != nil {
			t.Fatal(err)
		}
		name, got, err := encodeWithMethod(method, sliceSize, numInputs, numRecovery, true)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if !bytes.Equal(want[i], got[i]) {
				// Only the auto-selected kernel is a supported configuration;
				// forced kernels may be stubbed in this build, so just record it.
				if method == GF16Auto {
					t.Errorf("kernel %q sliceSize=%d: recovery block %d differs from Lookup reference", name, sliceSize, i)
				} else {
					fmt.Printf("note: kernel %q sliceSize=%d: recovery block %d differs from Lookup reference\n", name, sliceSize, i)
				}
			}
		}
	}
}

// TestEveryAvailableKernel forces each GF16 method the running CPU supports,
// in a separate process each, so that a crash inside one SIMD kernel is
// attributed to that kernel instead of taking the whole test binary down.
// Methods the CPU lacks fall back to the auto kernel inside ParPar and are
// skipped once the auto kernel itself has been exercised.
func TestEveryAvailableKernel(t *testing.T) {
	if os.Getenv(kernelEnv) != "" {
		t.Skip("subprocess")
	}
	autoName := ""
	seen := map[string]bool{}
	for method := GF16Auto; method <= GF16ClmulRVV; method++ {
		if !nativeMethod(int(method)) {
			continue
		}
		// Everything, including the probe, runs in a child: forcing a kernel
		// that is compiled out abort()s inside parpar_gfproc_new.
		cmd := exec.Command(os.Args[0], "-test.run=^TestKernelSubprocess$", "-test.v")
		cmd.Env = append(os.Environ(), kernelEnv+"="+strconv.Itoa(int(method)))
		out, runErr := cmd.CombinedOutput()
		name := reportedKernel(out)
		if method == GF16Auto {
			autoName = name
			t.Logf("auto-selected kernel: %q", autoName)
		}
		switch {
		case isIllegalInstruction(runErr):
			// ParPar trusts a forced method id, so a kernel the CPU cannot
			// execute dies with SIGILL rather than falling back.
			t.Logf("method %d (%s): not supported by this CPU", method, name)
			continue
		case name == "":
			t.Logf("method %d: not available in this build (%v)", method, runErr)
			continue
		case method != GF16Auto && autoName != "" && name == autoName:
			continue // unsupported on this CPU; ParPar fell back to the auto kernel
		case seen[name]:
			continue
		}
		seen[name] = true
		t.Run(fmt.Sprintf("%02d_%s", method, sanitize(name)), func(t *testing.T) {
			if runErr != nil {
				t.Errorf("kernel %q (method %d) failed: %v\n%s", name, method, runErr, tail(out, 2500))
			}
		})
	}
}

// reportedKernel extracts the kernel name from the first geometry line the
// child printed, or "" if it died before creating an encoder.
var kernelLine = regexp.MustCompile(`^kernel="((?:[^"\\]|\\.)*)"`)

func reportedKernel(out []byte) string {
	for _, line := range strings.Split(string(out), "\n") {
		if m := kernelLine.FindStringSubmatch(line); m != nil {
			if q, err := strconv.Unquote(`"` + m[1] + `"`); err == nil {
				return q
			}
		}
	}
	return ""
}

// nativeMethod reports whether a GF16 method id belongs to the running CPU
// architecture. Forcing a foreign-architecture id makes ParPar report that
// kernel's name while running a generic fallback with mismatched geometry,
// which is not a configuration any caller can reach through auto-detection.
func nativeMethod(method int) bool {
	switch method {
	case GF16Auto, GF16Lookup, GF16Lookup3:
		return true
	}
	x86 := method == GF16LookupSSE2 || (method >= GF16ShuffleSSSE3 && method <= GF16Affine2xAVX512)
	switch runtime.GOARCH {
	case "amd64", "386":
		return x86
	case "arm64", "arm":
		return !x86 && method != GF16Shuffle128RVV && method != GF16ClmulRVV
	case "riscv64":
		return method == GF16Shuffle128RVV || method == GF16ClmulRVV
	}
	return false
}

func isIllegalInstruction(err error) bool {
	var ee *exec.ExitError
	if !errors.As(err, &ee) {
		return false
	}
	if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() && ws.Signal() == syscall.SIGILL {
		return true
	}
	// Windows reports STATUS_ILLEGAL_INSTRUCTION as the exit code.
	return ee.ExitCode() == -1073741795 || uint32(ee.ExitCode()) == 0xC000001D
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			return r
		}
		return '_'
	}, s)
}

func tail(b []byte, n int) string {
	if len(b) > n {
		b = b[len(b)-n:]
	}
	return string(b)
}
