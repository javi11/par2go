package par2creator

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/javi11/par2go/internal/gf16"
	"github.com/javi11/par2go/internal/packets"
	"github.com/javi11/par2go/internal/rsenc"
)

func TestCreateBasic(t *testing.T) {
	// Create a temp directory
	tmpDir := t.TempDir()

	// Create a test input file
	inputPath := filepath.Join(tmpDir, "testfile.bin")
	data := make([]byte, 100000) // 100KB
	for i := range data {
		data[i] = byte(i % 256)
	}
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		t.Fatal(err)
	}

	outputPath := filepath.Join(tmpDir, "testfile.bin.par2")

	opts := Options{
		SliceSize:   10000, // 10KB slices → 10 input slices
		NumRecovery: 3,
		Creator:     "TestCreator",
	}

	err := Create(context.Background(), outputPath, []string{inputPath}, opts)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Check that main .par2 file was created
	if _, err := os.Stat(outputPath); err != nil {
		t.Fatalf("main .par2 file not created: %v", err)
	}

	// Check main file has content
	mainInfo, _ := os.Stat(outputPath)
	if mainInfo.Size() == 0 {
		t.Error("main .par2 file is empty")
	}

	// Check that volume files were created
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

	var volFiles []string
	for _, e := range entries {
		if strings.Contains(e.Name(), ".vol") && strings.HasSuffix(e.Name(), ".par2") {
			volFiles = append(volFiles, e.Name())
		}
	}

	if len(volFiles) == 0 {
		t.Error("no volume files created")
	}

	t.Logf("Created %d volume files: %v", len(volFiles), volFiles)

	// Volume files should have recovery data
	for _, vf := range volFiles {
		info, err := os.Stat(filepath.Join(tmpDir, vf))
		if err != nil {
			t.Errorf("stat %s: %v", vf, err)
			continue
		}
		if info.Size() == 0 {
			t.Errorf("volume file %s is empty", vf)
		}
	}
}

func TestCreateMultipleFiles(t *testing.T) {
	tmpDir := t.TempDir()

	// Create two input files
	files := []string{
		filepath.Join(tmpDir, "file1.bin"),
		filepath.Join(tmpDir, "file2.bin"),
	}

	for i, path := range files {
		data := make([]byte, 50000)
		for j := range data {
			data[j] = byte((i + j) % 256)
		}
		if err := os.WriteFile(path, data, 0644); err != nil {
			t.Fatal(err)
		}
	}

	outputPath := filepath.Join(tmpDir, "output.par2")

	opts := Options{
		SliceSize:   10000,
		NumRecovery: 4,
	}

	err := Create(context.Background(), outputPath, files, opts)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	if _, err := os.Stat(outputPath); err != nil {
		t.Fatalf("main .par2 file not created: %v", err)
	}
}

func TestCreateProgress(t *testing.T) {
	tmpDir := t.TempDir()

	inputPath := filepath.Join(tmpDir, "testfile.bin")
	data := make([]byte, 100000)
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		t.Fatal(err)
	}

	outputPath := filepath.Join(tmpDir, "testfile.bin.par2")

	phases := make(map[string]bool)
	var lastPct float64

	opts := Options{
		SliceSize:   10000,
		NumRecovery: 3,
		OnProgress: func(phase string, pct float64) {
			phases[phase] = true
			if pct < 0 || pct > 1.01 {
				t.Errorf("progress out of range: phase=%s pct=%f", phase, pct)
			}
			lastPct = pct
		},
	}

	err := Create(context.Background(), outputPath, []string{inputPath}, opts)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	if !phases["hashing"] {
		t.Error("hashing phase not reported")
	}
	if !phases["encoding"] {
		t.Error("encoding phase not reported")
	}
	if !phases["writing"] {
		t.Error("writing phase not reported")
	}
	_ = lastPct
}

func TestCreateCancel(t *testing.T) {
	tmpDir := t.TempDir()

	inputPath := filepath.Join(tmpDir, "testfile.bin")
	data := make([]byte, 100000)
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		t.Fatal(err)
	}

	outputPath := filepath.Join(tmpDir, "testfile.bin.par2")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	opts := Options{
		SliceSize:   10000,
		NumRecovery: 3,
	}

	err := Create(ctx, outputPath, []string{inputPath}, opts)
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

func TestCreateInvalidOptions(t *testing.T) {
	tmpDir := t.TempDir()
	inputPath := filepath.Join(tmpDir, "testfile.bin")
	if err := os.WriteFile(inputPath, []byte("test"), 0644); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(tmpDir, "test.par2")

	tests := []struct {
		name string
		opts Options
	}{
		{"zero slice size", Options{SliceSize: 0, NumRecovery: 1}},
		{"odd slice size", Options{SliceSize: 3, NumRecovery: 1}},
		{"zero recovery", Options{SliceSize: 4, NumRecovery: 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Create(context.Background(), outputPath, []string{inputPath}, tt.opts)
			if err == nil {
				t.Error("expected error for invalid options")
			}
		})
	}
}

func TestCreateNoInputFiles(t *testing.T) {
	opts := Options{SliceSize: 1000, NumRecovery: 1}
	err := Create(context.Background(), "/tmp/test.par2", nil, opts)
	if err == nil {
		t.Error("expected error for no input files")
	}
}

func BenchmarkCreate1MB(b *testing.B) {
	for i := 0; i < b.N; i++ {
		tmpDir := b.TempDir()
		inputPath := filepath.Join(tmpDir, "testfile.bin")
		data := make([]byte, 1024*1024)
		for j := range data {
			data[j] = byte(j)
		}
		if err := os.WriteFile(inputPath, data, 0644); err != nil {
			b.Fatal(err)
		}

		outputPath := filepath.Join(tmpDir, "testfile.bin.par2")
		opts := Options{
			SliceSize:   10000,
			NumRecovery: 10,
		}
		_ = Create(context.Background(), outputPath, []string{inputPath}, opts)
	}
}

func TestVolumeFileDoublingStrategy(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a file large enough to produce many recovery blocks
	inputPath := filepath.Join(tmpDir, "testfile.bin")
	data := make([]byte, 100000)
	for i := range data {
		data[i] = byte(i)
	}
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		t.Fatal(err)
	}

	outputPath := filepath.Join(tmpDir, "testfile.bin.par2")

	// 15 recovery blocks should produce volumes: 1, 1, 2, 4, 8 (but last may be truncated to 7)
	opts := Options{
		SliceSize:   10000,
		NumRecovery: 15,
	}

	err := Create(context.Background(), outputPath, []string{inputPath}, opts)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

	var volFiles []string
	for _, e := range entries {
		if strings.Contains(e.Name(), ".vol") && strings.HasSuffix(e.Name(), ".par2") {
			volFiles = append(volFiles, e.Name())
			t.Logf("Volume file: %s", e.Name())
		}
	}

	if len(volFiles) == 0 {
		t.Error("no volume files created")
	}
}

// --- Integration test helpers ---

// parsedPacket represents a parsed PAR2 packet for test validation.
type parsedPacket struct {
	Magic         [8]byte
	Length        uint64
	PacketHash    [16]byte
	RecoverySetID [16]byte
	Type          [16]byte
	Body          []byte
}

// readPAR2Packets reads a PAR2 file and returns all parsed packets.
// It verifies each packet's magic bytes and hash integrity.
func readPAR2Packets(t *testing.T, path string) []parsedPacket {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}

	var pkts []parsedPacket
	offset := 0

	for offset < len(data) {
		if offset+packets.HeaderSize > len(data) {
			t.Fatalf("truncated packet header at offset %d in %s", offset, path)
		}

		var pkt parsedPacket
		copy(pkt.Magic[:], data[offset:offset+8])
		pkt.Length = binary.LittleEndian.Uint64(data[offset+8 : offset+16])
		copy(pkt.PacketHash[:], data[offset+16:offset+32])
		copy(pkt.RecoverySetID[:], data[offset+32:offset+48])
		copy(pkt.Type[:], data[offset+48:offset+64])

		if pkt.Magic != packets.Magic {
			t.Fatalf("bad magic at offset %d: %x", offset, pkt.Magic)
		}

		bodyLen := int(pkt.Length) - packets.HeaderSize
		if bodyLen < 0 || offset+int(pkt.Length) > len(data) {
			t.Fatalf("invalid packet length %d at offset %d", pkt.Length, offset)
		}

		pkt.Body = data[offset+packets.HeaderSize : offset+int(pkt.Length)]

		// Verify packet hash: MD5(RecoverySetID + Type + Body)
		h := md5.New()
		h.Write(pkt.RecoverySetID[:])
		h.Write(pkt.Type[:])
		h.Write(pkt.Body)
		var expectedHash [16]byte
		copy(expectedHash[:], h.Sum(nil))

		if pkt.PacketHash != expectedHash {
			t.Errorf("packet hash mismatch at offset %d (type %x)", offset, pkt.Type)
		}

		pkts = append(pkts, pkt)
		offset += int(pkt.Length)
	}

	return pkts
}

// --- Integration tests ---

func TestCreateAndValidatePackets(t *testing.T) {
	tmpDir := t.TempDir()
	sliceSize := 1000

	// Create two input files with deterministic content
	file1Path := filepath.Join(tmpDir, "alpha.bin")
	file1Data := make([]byte, 2500)
	for i := range file1Data {
		file1Data[i] = byte(i*7 + 3)
	}
	if err := os.WriteFile(file1Path, file1Data, 0644); err != nil {
		t.Fatal(err)
	}

	file2Path := filepath.Join(tmpDir, "bravo.bin")
	file2Data := make([]byte, 1500)
	for i := range file2Data {
		file2Data[i] = byte(i*13 + 5)
	}
	if err := os.WriteFile(file2Path, file2Data, 0644); err != nil {
		t.Fatal(err)
	}

	outputPath := filepath.Join(tmpDir, "output.par2")
	opts := Options{
		SliceSize:   sliceSize,
		NumRecovery: 2,
		Creator:     "IntegrationTest",
	}

	err := Create(context.Background(), outputPath, []string{file1Path, file2Path}, opts)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// --- Parse the main .par2 file ---
	pkts := readPAR2Packets(t, outputPath)

	// Count packet types
	var mainPkts, creatorPkts, fdPkts, ifscPkts, recoveryPkts int
	for _, pkt := range pkts {
		switch pkt.Type {
		case packets.TypeMain:
			mainPkts++
		case packets.TypeCreator:
			creatorPkts++
		case packets.TypeFileDescription:
			fdPkts++
		case packets.TypeIFSC:
			ifscPkts++
		case packets.TypeRecoverySlice:
			recoveryPkts++
		}
	}

	// 1. Packet count: 1 Main + 1 Creator + 2 FileDesc + 2 IFSC = 6
	if got, want := len(pkts), 6; got != want {
		t.Errorf("packet count: got %d, want %d (main=%d creator=%d fd=%d ifsc=%d recovery=%d)",
			got, want, mainPkts, creatorPkts, fdPkts, ifscPkts, recoveryPkts)
	}
	if mainPkts != 1 {
		t.Errorf("main packets: got %d, want 1", mainPkts)
	}
	if creatorPkts != 1 {
		t.Errorf("creator packets: got %d, want 1", creatorPkts)
	}
	if fdPkts != 2 {
		t.Errorf("file description packets: got %d, want 2", fdPkts)
	}
	if ifscPkts != 2 {
		t.Errorf("IFSC packets: got %d, want 2", ifscPkts)
	}
	if recoveryPkts != 0 {
		t.Errorf("recovery packets in main file: got %d, want 0", recoveryPkts)
	}

	// 2. Validate Main packet
	var mainPkt parsedPacket
	for _, pkt := range pkts {
		if pkt.Type == packets.TypeMain {
			mainPkt = pkt
			break
		}
	}

	mainSliceSize := binary.LittleEndian.Uint64(mainPkt.Body[0:8])
	if mainSliceSize != uint64(sliceSize) {
		t.Errorf("main packet sliceSize: got %d, want %d", mainSliceSize, sliceSize)
	}

	recoverableCount := binary.LittleEndian.Uint32(mainPkt.Body[8:12])
	if recoverableCount != 2 {
		t.Errorf("recoverable file count: got %d, want 2", recoverableCount)
	}

	// File IDs must be sorted (each ≤ next by bytes.Compare)
	if recoverableCount >= 2 {
		id1 := mainPkt.Body[12:28]
		id2 := mainPkt.Body[28:44]
		if bytes.Compare(id1, id2) > 0 {
			t.Error("file IDs in Main packet are not sorted")
		}
	}

	// 3. Recovery Set ID consistent across all packets
	refSetID := pkts[0].RecoverySetID
	for i, pkt := range pkts[1:] {
		if pkt.RecoverySetID != refSetID {
			t.Errorf("packet %d has different recovery set ID", i+1)
		}
	}

	// 4. Validate File Description packets
	type fdInfo struct {
		fileSize uint64
		filename string
		hash16k  [16]byte
		hashFull [16]byte
	}
	var fds []fdInfo
	for _, pkt := range pkts {
		if pkt.Type != packets.TypeFileDescription {
			continue
		}
		body := pkt.Body
		var fd fdInfo
		copy(fd.hashFull[:], body[16:32])
		copy(fd.hash16k[:], body[32:48])
		fd.fileSize = binary.LittleEndian.Uint64(body[48:56])
		fd.filename = strings.TrimRight(string(body[56:]), "\x00")
		fds = append(fds, fd)
	}

	expectedFiles := map[string]uint64{
		"alpha.bin": 2500,
		"bravo.bin": 1500,
	}
	for _, fd := range fds {
		expectedSize, ok := expectedFiles[fd.filename]
		if !ok {
			t.Errorf("unexpected file description for %q", fd.filename)
			continue
		}
		if fd.fileSize != expectedSize {
			t.Errorf("file %s size: got %d, want %d", fd.filename, fd.fileSize, expectedSize)
		}
		var zeroHash [16]byte
		if fd.hash16k == zeroHash {
			t.Errorf("file %s hash16k is zero", fd.filename)
		}
		if fd.hashFull == zeroHash {
			t.Errorf("file %s hashFull is zero", fd.filename)
		}
	}

	// 5. Validate IFSC packets — slice count matches ceil(fileSize / sliceSize)
	for _, pkt := range pkts {
		if pkt.Type != packets.TypeIFSC {
			continue
		}
		body := pkt.Body
		fileID := body[0:16]
		numEntries := (len(body) - 16) / 20 // each entry: 16 MD5 + 4 CRC32

		for _, fd := range fds {
			fid := packets.FileID(fd.hash16k, fd.fileSize, fd.filename)
			if !bytes.Equal(fid[:], fileID) {
				continue
			}
			expectedSlices := int((fd.fileSize + uint64(sliceSize) - 1) / uint64(sliceSize))
			if numEntries != expectedSlices {
				t.Errorf("IFSC for %s: got %d entries, want %d", fd.filename, numEntries, expectedSlices)
			}
		}
	}

	// 6. Validate Creator packet
	for _, pkt := range pkts {
		if pkt.Type != packets.TypeCreator {
			continue
		}
		creatorStr := strings.TrimRight(string(pkt.Body), "\x00")
		if creatorStr != "IntegrationTest" {
			t.Errorf("creator string: got %q, want %q", creatorStr, "IntegrationTest")
		}
	}

	// 7. Validate volume files contain RecoverySlice packets
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

	var volFilePaths []string
	for _, e := range entries {
		if strings.Contains(e.Name(), ".vol") && strings.HasSuffix(e.Name(), ".par2") {
			volFilePaths = append(volFilePaths, filepath.Join(tmpDir, e.Name()))
		}
	}
	if len(volFilePaths) == 0 {
		t.Fatal("no volume files created")
	}

	totalRecoveryPkts := 0
	for _, vf := range volFilePaths {
		info, err := os.Stat(vf)
		if err != nil {
			t.Errorf("stat %s: %v", vf, err)
			continue
		}
		if info.Size() == 0 {
			t.Errorf("volume file %s is empty", vf)
			continue
		}

		volPkts := readPAR2Packets(t, vf)
		for _, pkt := range volPkts {
			if pkt.Type != packets.TypeRecoverySlice {
				t.Errorf("volume file %s contains non-recovery packet type %x", vf, pkt.Type)
			}
			// Recovery slice body: 4 bytes exponent + sliceSize bytes data
			if got, want := len(pkt.Body), 4+sliceSize; got != want {
				t.Errorf("recovery slice body size: got %d, want %d", got, want)
			}
			totalRecoveryPkts++
		}
	}
	if totalRecoveryPkts != 2 {
		t.Errorf("total recovery packets across volumes: got %d, want 2", totalRecoveryPkts)
	}
}

func TestCreateAndRecoverCorruptedSlice(t *testing.T) {
	tmpDir := t.TempDir()

	sliceSize := 100 // multiple of 4, even (required for GF16)
	fileSize := 500  // exactly 5 slices
	numRecovery := 2

	// Create input file with deterministic content
	inputPath := filepath.Join(tmpDir, "data.bin")
	originalData := make([]byte, fileSize)
	for i := range originalData {
		originalData[i] = byte(i*17 + 11)
	}
	if err := os.WriteFile(inputPath, originalData, 0644); err != nil {
		t.Fatal(err)
	}

	outputPath := filepath.Join(tmpDir, "data.bin.par2")
	opts := Options{
		SliceSize:   sliceSize,
		NumRecovery: numRecovery,
	}

	err := Create(context.Background(), outputPath, []string{inputPath}, opts)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// Parse volume files to extract recovery data
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

	type recoveryInfo struct {
		exponent uint16
		data     []byte
	}
	var recoveries []recoveryInfo

	for _, e := range entries {
		if !strings.Contains(e.Name(), ".vol") || !strings.HasSuffix(e.Name(), ".par2") {
			continue
		}
		volPkts := readPAR2Packets(t, filepath.Join(tmpDir, e.Name()))
		for _, pkt := range volPkts {
			if pkt.Type == packets.TypeRecoverySlice {
				exp := binary.LittleEndian.Uint32(pkt.Body[0:4])
				data := make([]byte, len(pkt.Body)-4)
				copy(data, pkt.Body[4:])
				recoveries = append(recoveries, recoveryInfo{
					exponent: uint16(exp),
					data:     data,
				})
			}
		}
	}
	if len(recoveries) == 0 {
		t.Fatal("no recovery blocks found in volume files")
	}

	numInputSlices := (fileSize + sliceSize - 1) / sliceSize
	constants := rsenc.GenerateConstants(numInputSlices)

	// getSlice returns the i-th input slice, zero-padded to sliceSize.
	getSlice := func(i int) []byte {
		buf := make([]byte, sliceSize)
		start := i * sliceSize
		end := start + sliceSize
		if end > fileSize {
			end = fileSize
		}
		copy(buf, originalData[start:end])
		return buf
	}

	// Recover a corrupted slice using each available recovery block.
	// This validates the full encode→parse→recover pipeline.
	corruptedIdx := 2 // lose the 3rd slice

	for _, rec := range recoveries {
		t.Run(
			func() string {
				if rec.exponent == 0 {
					return "exponent_0"
				}
				return "exponent_1"
			}(),
			func(t *testing.T) {
				// Recovery math:
				//   recovery[e] = XOR_i( input[i] * pow(constants[i], exponent) )
				// Isolate lost slice j:
				//   intermediate = recovery[e] XOR sum_{i!=j}( input[i] * pow(constants[i], e) )
				//                = input[j] * pow(constants[j], e)
				//   recovered   = intermediate * inv( pow(constants[j], e) )
				intermediate := make([]byte, sliceSize)
				copy(intermediate, rec.data)

				for i := 0; i < numInputSlices; i++ {
					if i == corruptedIdx {
						continue
					}
					factor := gf16.Pow(constants[i], rec.exponent)
					gf16.MulAccumulate(intermediate, getSlice(i), factor)
				}

				corruptedFactor := gf16.Pow(constants[corruptedIdx], rec.exponent)
				invFactor := gf16.Inv(corruptedFactor)
				recovered := make([]byte, sliceSize)
				gf16.MulAccumulate(recovered, intermediate, invFactor)

				originalSlice := getSlice(corruptedIdx)
				if !bytes.Equal(recovered, originalSlice) {
					diffCount := 0
					for i := range recovered {
						if recovered[i] != originalSlice[i] {
							if diffCount < 5 {
								t.Errorf("byte %d: got 0x%02x, want 0x%02x", i, recovered[i], originalSlice[i])
							}
							diffCount++
						}
					}
					t.Errorf("recovered slice differs from original in %d/%d bytes", diffCount, sliceSize)
				}
			},
		)
	}
}
