package par2go

import (
	"runtime"
	"strings"
	"testing"

	"github.com/javi11/par2go/internal/parpar"
	"golang.org/x/sys/cpu"
)

// skipIfCPUMismatch skips t when parpar's auto-selected GF16 method requires
// CPU features (GFNI, AVX-512) that golang.org/x/sys/cpu reports as absent.
// This guards against VM/hypervisor environments that advertise the CPU flags
// in CPUID but fault on the actual instructions at runtime.
// Querying the method name via MethodName() is safe – only encoding crashes.
func skipIfCPUMismatch(t *testing.T) {
	t.Helper()
	if runtime.GOARCH != "amd64" {
		return
	}
	proc, err := parpar.NewGfProc(64, 0)
	if err != nil || proc == nil {
		t.Skip("cannot initialise GfProc")
		return
	}
	name := proc.MethodName()
	proc.Close()
	if strings.Contains(name, "GFNI") && !cpu.X86.HasAVX512GFNI {
		t.Skipf("parpar selected %q but CPU reports no GFNI support; skipping encode test", name)
	}
	if strings.Contains(name, "AVX512") && !cpu.X86.HasAVX512F {
		t.Skipf("parpar selected %q but CPU reports no AVX-512 support; skipping encode test", name)
	}
}
