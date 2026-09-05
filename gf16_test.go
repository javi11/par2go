package par2go

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/javi11/par2go/internal/parpar"
)

// Downstream applications set Options.Method to pin a kernel. The values must
// be the bridge's PARPAR_GF16_* constants, so a drift here would silently
// select a different kernel.
func TestGF16MethodConstantsMatchBridge(t *testing.T) {
	pairs := map[string][2]int{
		"Auto":           {GF16Auto, parpar.GF16Auto},
		"Lookup":         {GF16Lookup, parpar.GF16Lookup},
		"Lookup3":        {GF16Lookup3, parpar.GF16Lookup3},
		"ShuffleAVX2":    {GF16ShuffleAVX2, parpar.GF16ShuffleAVX2},
		"ShuffleAVX512":  {GF16ShuffleAVX512, parpar.GF16ShuffleAVX512},
		"ShuffleVBMI":    {GF16ShuffleVBMI, parpar.GF16ShuffleVBMI},
		"XorJitAVX2":     {GF16XorJitAVX2, parpar.GF16XorJitAVX2},
		"XorJitAVX512":   {GF16XorJitAVX512, parpar.GF16XorJitAVX512},
		"AffineGFNI":     {GF16AffineGFNI, parpar.GF16AffineGFNI},
		"AffineAVX2":     {GF16AffineAVX2, parpar.GF16AffineAVX2},
		"AffineAVX512":   {GF16AffineAVX512, parpar.GF16AffineAVX512},
		"Affine2xAVX512": {GF16Affine2xAVX512, parpar.GF16Affine2xAVX512},
		"ShuffleNEON":    {GF16ShuffleNEON, parpar.GF16ShuffleNEON},
		"ClmulNEON":      {GF16ClmulNEON, parpar.GF16ClmulNEON},
		"ClmulSHA3":      {GF16ClmulSHA3, parpar.GF16ClmulSHA3},
		"Shuffle128SVE2": {GF16Shuffle128SVE2, parpar.GF16Shuffle128SVE2},
	}
	for name, p := range pairs {
		if p[0] != p[1] {
			t.Errorf("GF16%s = %d, bridge has %d", name, p[0], p[1])
		}
	}
}

// Pinning the portable Lookup kernel through the public API must yield the
// same PAR2 set as auto-selection: a downstream app that pins a kernel to
// dodge a broken one must not get different recovery data.
func TestCreateWithPinnedKernelMatchesAuto(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "input.bin")
	data := make([]byte, 100_000)
	for i := range data {
		data[i] = byte(i*7 + i>>8)
	}
	if err := os.WriteFile(input, data, 0o644); err != nil {
		t.Fatal(err)
	}

	create := func(sub string, method int) map[string][]byte {
		out := filepath.Join(dir, sub)
		if err := os.MkdirAll(out, 0o755); err != nil {
			t.Fatal(err)
		}
		opts := Options{SliceSize: 9984, NumRecovery: 3, NumGoroutines: 2, Method: method}
		if err := Create(context.Background(), filepath.Join(out, "input.par2"), []string{input}, opts); err != nil {
			t.Fatalf("Create(method=%d): %v", method, err)
		}
		return readPar2Files(t, out)
	}

	auto := create("auto", GF16Auto)
	pinned := create("pinned", GF16Lookup)
	if len(auto) == 0 || len(auto) != len(pinned) {
		t.Fatalf("expected the same PAR2 file set, got %d vs %d files", len(auto), len(pinned))
	}
	for name, want := range auto {
		if got, ok := pinned[name]; !ok || !bytes.Equal(got, want) {
			t.Errorf("%s differs between auto-selected and pinned Lookup kernel", name)
		}
	}
}
