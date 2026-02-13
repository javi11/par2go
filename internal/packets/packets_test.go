package packets

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"hash/crc32"
	"testing"
)

func TestWriteHeader(t *testing.T) {
	recoverySetID := [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	body := []byte("test body data")

	var buf bytes.Buffer
	err := WriteHeader(&buf, recoverySetID, TypeMain, body)
	if err != nil {
		t.Fatalf("WriteHeader failed: %v", err)
	}

	data := buf.Bytes()

	// Total size = 64 header + body
	expectedLen := HeaderSize + len(body)
	if len(data) != expectedLen {
		t.Fatalf("expected %d bytes, got %d", expectedLen, len(data))
	}

	// Check magic
	if !bytes.Equal(data[0:8], Magic[:]) {
		t.Error("magic bytes mismatch")
	}

	// Check length
	length := binary.LittleEndian.Uint64(data[8:16])
	if length != uint64(expectedLen) {
		t.Errorf("length field: got %d, want %d", length, expectedLen)
	}

	// Check recovery set ID
	if !bytes.Equal(data[32:48], recoverySetID[:]) {
		t.Error("recovery set ID mismatch")
	}

	// Check type
	if !bytes.Equal(data[48:64], TypeMain[:]) {
		t.Error("packet type mismatch")
	}

	// Check body
	if !bytes.Equal(data[64:], body) {
		t.Error("body mismatch")
	}

	// Verify packet hash: MD5(recoverySetID + type + body)
	h := md5.New()
	h.Write(recoverySetID[:])
	h.Write(TypeMain[:])
	h.Write(body)
	expectedHash := h.Sum(nil)
	if !bytes.Equal(data[16:32], expectedHash) {
		t.Error("packet hash mismatch")
	}
}

func TestFileID(t *testing.T) {
	hash16k := [16]byte{0xAA, 0xBB, 0xCC, 0xDD}
	fileSize := uint64(12345)
	name := "testfile.bin"

	id := FileID(hash16k, fileSize, name)

	// Verify deterministic
	id2 := FileID(hash16k, fileSize, name)
	if id != id2 {
		t.Error("FileID not deterministic")
	}

	// Different name should produce different ID
	id3 := FileID(hash16k, fileSize, "other.bin")
	if id == id3 {
		t.Error("different name should produce different FileID")
	}

	// Different size should produce different ID
	id4 := FileID(hash16k, 99999, name)
	if id == id4 {
		t.Error("different size should produce different FileID")
	}
}

func TestMainPacket(t *testing.T) {
	sliceSize := uint64(10000)
	fileIDs := [][16]byte{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	}

	body := MainPacket(sliceSize, fileIDs)

	// Expected: 8 (sliceSize) + 4 (numRecoveryFiles) + 2*16 (fileIDs) = 44
	expectedLen := 8 + 4 + 2*16
	if len(body) != expectedLen {
		t.Fatalf("expected body length %d, got %d", expectedLen, len(body))
	}

	// Check sliceSize
	gotSliceSize := binary.LittleEndian.Uint64(body[0:8])
	if gotSliceSize != sliceSize {
		t.Errorf("sliceSize: got %d, want %d", gotSliceSize, sliceSize)
	}

	// Check number of recoverable files = len(fileIDs)
	numRecFiles := binary.LittleEndian.Uint32(body[8:12])
	if numRecFiles != 2 {
		t.Errorf("number of recoverable files: got %d, want 2", numRecFiles)
	}

	// Check file IDs
	if !bytes.Equal(body[12:28], fileIDs[0][:]) {
		t.Error("first file ID mismatch")
	}
	if !bytes.Equal(body[28:44], fileIDs[1][:]) {
		t.Error("second file ID mismatch")
	}
}

func TestRecoverySetID(t *testing.T) {
	mainBody := []byte("test main body")

	rsID := RecoverySetID(mainBody)
	expected := md5.Sum(mainBody)

	if rsID != expected {
		t.Error("RecoverySetID should be MD5 of main body")
	}
}

func TestFileDescriptionPacket(t *testing.T) {
	fileID := [16]byte{1, 2, 3}
	hashFull := [16]byte{4, 5, 6}
	hash16k := [16]byte{7, 8, 9}
	fileSize := uint64(12345)
	name := "test.bin"

	body := FileDescriptionPacket(fileID, hashFull, hash16k, fileSize, name)

	// Check fileID
	if !bytes.Equal(body[0:16], fileID[:]) {
		t.Error("fileID mismatch")
	}

	// Check hashFull
	if !bytes.Equal(body[16:32], hashFull[:]) {
		t.Error("hashFull mismatch")
	}

	// Check hash16k
	if !bytes.Equal(body[32:48], hash16k[:]) {
		t.Error("hash16k mismatch")
	}

	// Check fileSize
	gotSize := binary.LittleEndian.Uint64(body[48:56])
	if gotSize != fileSize {
		t.Errorf("fileSize: got %d, want %d", gotSize, fileSize)
	}

	// Check name is present and padded to 4-byte boundary
	nameStart := 56
	paddedLen := (len(name) + 3) &^ 3
	if len(body) != nameStart+paddedLen {
		t.Errorf("body length: got %d, want %d", len(body), nameStart+paddedLen)
	}

	// Verify name content
	gotName := string(bytes.TrimRight(body[nameStart:], "\x00"))
	if gotName != name {
		t.Errorf("name: got %q, want %q", gotName, name)
	}
}

func TestIFSCPacket(t *testing.T) {
	fileID := [16]byte{1, 2, 3}
	entries := []IFSCEntry{
		{MD5: [16]byte{10, 11, 12}, CRC32: 0xDEADBEEF},
		{MD5: [16]byte{20, 21, 22}, CRC32: 0xCAFEBABE},
	}

	body := IFSCPacket(fileID, entries)

	// Expected: 16 (fileID) + 2 * 20 (entries) = 56
	expectedLen := 16 + 2*20
	if len(body) != expectedLen {
		t.Fatalf("body length: got %d, want %d", len(body), expectedLen)
	}

	// Check fileID
	if !bytes.Equal(body[0:16], fileID[:]) {
		t.Error("fileID mismatch")
	}

	// Check first entry
	if !bytes.Equal(body[16:32], entries[0].MD5[:]) {
		t.Error("entry 0 MD5 mismatch")
	}
	gotCRC := binary.LittleEndian.Uint32(body[32:36])
	if gotCRC != entries[0].CRC32 {
		t.Errorf("entry 0 CRC32: got %x, want %x", gotCRC, entries[0].CRC32)
	}

	// Check second entry
	if !bytes.Equal(body[36:52], entries[1].MD5[:]) {
		t.Error("entry 1 MD5 mismatch")
	}
	gotCRC = binary.LittleEndian.Uint32(body[52:56])
	if gotCRC != entries[1].CRC32 {
		t.Errorf("entry 1 CRC32: got %x, want %x", gotCRC, entries[1].CRC32)
	}
}

func TestRecoverySlicePacket(t *testing.T) {
	exponent := uint16(42)
	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}

	body := RecoverySlicePacket(exponent, data)

	// Expected: 4 (exponent as uint32) + 8 (data) = 12
	if len(body) != 12 {
		t.Fatalf("body length: got %d, want 12", len(body))
	}

	gotExp := binary.LittleEndian.Uint32(body[0:4])
	if gotExp != uint32(exponent) {
		t.Errorf("exponent: got %d, want %d", gotExp, exponent)
	}

	if !bytes.Equal(body[4:], data) {
		t.Error("recovery data mismatch")
	}
}

func TestCreatorPacket(t *testing.T) {
	creator := "Postie"
	body := CreatorPacket(creator)

	// Should be padded to 4-byte boundary
	paddedLen := (len(creator) + 3) &^ 3
	if len(body) != paddedLen {
		t.Fatalf("body length: got %d, want %d", len(body), paddedLen)
	}

	// Check content
	gotCreator := string(bytes.TrimRight(body, "\x00"))
	if gotCreator != creator {
		t.Errorf("creator: got %q, want %q", gotCreator, creator)
	}
}

func TestCreatorPacketPadding(t *testing.T) {
	tests := []struct {
		name        string
		creator     string
		expectedLen int
	}{
		{"4 chars", "Test", 4},
		{"5 chars", "Tests", 8},
		{"8 chars", "TestTest", 8},
		{"1 char", "T", 4},
		{"empty", "", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := CreatorPacket(tt.creator)
			if len(body) != tt.expectedLen {
				t.Errorf("length for %q: got %d, want %d", tt.creator, len(body), tt.expectedLen)
			}
		})
	}
}

func TestComputeSliceChecksums(t *testing.T) {
	data := make([]byte, 100)
	for i := range data {
		data[i] = byte(i)
	}
	sliceSize := 32

	entries := ComputeSliceChecksums(data, sliceSize)

	// 100 bytes / 32 = 4 slices (last one is 4 bytes, padded to 32)
	if len(entries) != 4 {
		t.Fatalf("expected 4 entries, got %d", len(entries))
	}

	// Verify first slice manually
	firstSlice := data[0:32]
	expectedMD5 := md5.Sum(firstSlice)
	expectedCRC := crc32.ChecksumIEEE(firstSlice)

	if entries[0].MD5 != expectedMD5 {
		t.Error("first slice MD5 mismatch")
	}
	if entries[0].CRC32 != expectedCRC {
		t.Errorf("first slice CRC32: got %x, want %x", entries[0].CRC32, expectedCRC)
	}

	// Verify last slice is padded
	lastSlice := make([]byte, 32)
	copy(lastSlice, data[96:100])
	expectedMD5 = md5.Sum(lastSlice)
	expectedCRC = crc32.ChecksumIEEE(lastSlice)

	if entries[3].MD5 != expectedMD5 {
		t.Error("last slice MD5 mismatch (padding issue?)")
	}
	if entries[3].CRC32 != expectedCRC {
		t.Errorf("last slice CRC32: got %x, want %x", entries[3].CRC32, expectedCRC)
	}
}

func TestWriteHeaderAllTypes(t *testing.T) {
	// Verify all packet types can be written
	types := [][16]byte{
		TypeMain,
		TypeFileDescription,
		TypeIFSC,
		TypeRecoverySlice,
		TypeCreator,
	}

	recoverySetID := [16]byte{}
	body := []byte("test")

	for _, pt := range types {
		var buf bytes.Buffer
		err := WriteHeader(&buf, recoverySetID, pt, body)
		if err != nil {
			t.Errorf("WriteHeader failed for type %v: %v", pt, err)
		}
		if buf.Len() != HeaderSize+len(body) {
			t.Errorf("wrong size for type %v: got %d", pt, buf.Len())
		}
	}
}
