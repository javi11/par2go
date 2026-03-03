package parpar

import (
	"bytes"
	"testing"
)

func TestNewGfProc(t *testing.T) {
	proc, err := NewGfProc(1024, 0)
	if err != nil {
		t.Fatalf("NewGfProc failed: %v", err)
	}
	defer proc.Close()

	name := proc.MethodName()
	if name == "" || name == "unknown" {
		t.Errorf("unexpected method name: %q", name)
	}
	t.Logf("Method: %s, Threads: %d", name, proc.NumThreads())
}

func TestGfProcBasicEncode(t *testing.T) {
	sliceSize := 1024
	numInputs := 5
	numRecovery := 3

	proc, err := NewGfProc(sliceSize, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()

	exponents := make([]uint16, numRecovery)
	for i := range exponents {
		exponents[i] = uint16(i)
	}
	proc.SetRecoverySlices(exponents)

	for i := 0; i < numInputs; i++ {
		data := make([]byte, sliceSize)
		for j := range data {
			data[j] = byte(i*7 + j)
		}
		proc.Add(i, data)
	}
	proc.End()

	outputs := make([][]byte, numRecovery)
	for i := range outputs {
		outputs[i] = make([]byte, sliceSize)
		proc.GetOutput(i, outputs[i])
	}
	proc.FreeMem()

	for i, out := range outputs {
		allZero := true
		for _, b := range out {
			if b != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("recovery block %d is all zeros", i)
		}
	}
}

func TestGfProcDeterministic(t *testing.T) {
	sliceSize := 512
	numInputs := 4
	numRecovery := 2

	encode := func() [][]byte {
		proc, err := NewGfProc(sliceSize, 1)
		if err != nil {
			t.Fatal(err)
		}
		defer proc.Close()

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

		results := make([][]byte, numRecovery)
		for i := range results {
			results[i] = make([]byte, sliceSize)
			proc.GetOutput(i, results[i])
		}
		return results
	}

	out1 := encode()
	out2 := encode()

	for i := range out1 {
		if !bytes.Equal(out1[i], out2[i]) {
			t.Errorf("recovery block %d: non-deterministic encoding", i)
		}
	}
}

func BenchmarkGfProcEncode(b *testing.B) {
	sliceSize := 768 * 1024 // 768KB — typical PAR2 slice
	numInputs := 20
	numRecovery := 5

	// Prepare input data once outside the loop
	inputs := make([][]byte, numInputs)
	for i := range inputs {
		inputs[i] = make([]byte, sliceSize)
		for j := range inputs[i] {
			inputs[i][j] = byte(i*7 + j)
		}
	}

	exponents := make([]uint16, numRecovery)
	for i := range exponents {
		exponents[i] = uint16(i)
	}

	b.SetBytes(int64(numInputs) * int64(sliceSize))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		proc, err := NewGfProc(sliceSize, 0)
		if err != nil {
			b.Fatal(err)
		}
		proc.SetRecoverySlices(exponents)
		for j, in := range inputs {
			proc.Add(j, in)
		}
		proc.End()
		dst := make([]byte, sliceSize)
		for j := range numRecovery {
			proc.GetOutput(j, dst)
		}
		proc.FreeMem()
		proc.Close()
	}
	b.Logf("Method: %s", func() string {
		p, _ := NewGfProc(sliceSize, 0)
		if p == nil {
			return "unknown"
		}
		n := p.MethodName()
		p.Close()
		return n
	}())
}
