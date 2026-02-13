// Package packets implements the PAR2 binary packet format per the PAR2 specification.
//
// Reference: http://parchive.sourceforge.net/docs/specifications/parity-volume-spec/article-spec.html
package packets

import (
	"crypto/md5"
	"encoding/binary"
	"hash/crc32"
	"io"
)

// Magic is the PAR2 packet magic string.
var Magic = [8]byte{'P', 'A', 'R', '2', '\x00', 'P', 'K', 'T'}

// Packet type signatures (16 bytes each).
var (
	TypeMain             = [16]byte{'P', 'A', 'R', ' ', '2', '.', '0', '\x00', 'M', 'a', 'i', 'n', '\x00', '\x00', '\x00', '\x00'}
	TypeFileDescription  = [16]byte{'P', 'A', 'R', ' ', '2', '.', '0', '\x00', 'F', 'i', 'l', 'e', 'D', 'e', 's', 'c'}
	TypeIFSC             = [16]byte{'P', 'A', 'R', ' ', '2', '.', '0', '\x00', 'I', 'F', 'S', 'C', '\x00', '\x00', '\x00', '\x00'}
	TypeRecoverySlice    = [16]byte{'P', 'A', 'R', ' ', '2', '.', '0', '\x00', 'R', 'e', 'c', 'v', 'S', 'l', 'i', 'c'}
	TypeCreator          = [16]byte{'P', 'A', 'R', ' ', '2', '.', '0', '\x00', 'C', 'r', 'e', 'a', 't', 'o', 'r', '\x00'}
)

// HeaderSize is the size of a PAR2 packet header in bytes.
const HeaderSize = 64

// Header represents a PAR2 packet header (64 bytes).
type Header struct {
	Magic         [8]byte  // "PAR2\x00PKT"
	Length        uint64   // Packet length (header + body)
	PacketHash   [16]byte // MD5 of (Recovery Set ID + Type + Body)
	RecoverySetID [16]byte // Links packets to the same recovery set
	Type          [16]byte // Packet type signature
}

// WriteHeader writes a PAR2 packet header to w.
// body contains the packet body (after the type field).
// The packet hash is computed over: RecoverySetID + Type + body.
func WriteHeader(w io.Writer, recoverySetID [16]byte, packetType [16]byte, body []byte) error {
	length := uint64(HeaderSize + len(body))

	// Compute packet hash: MD5(RecoverySetID + Type + body)
	h := md5.New()
	h.Write(recoverySetID[:])
	h.Write(packetType[:])
	h.Write(body)
	var packetHash [16]byte
	copy(packetHash[:], h.Sum(nil))

	// Write header fields
	if _, err := w.Write(Magic[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, length); err != nil {
		return err
	}
	if _, err := w.Write(packetHash[:]); err != nil {
		return err
	}
	if _, err := w.Write(recoverySetID[:]); err != nil {
		return err
	}
	if _, err := w.Write(packetType[:]); err != nil {
		return err
	}
	// Write body
	_, err := w.Write(body)
	return err
}

// FileID computes a PAR2 file ID from the file's 16K MD5 hash, file size, and name.
// fileID = MD5(hash16k + length + name_bytes)
func FileID(hash16k [16]byte, fileSize uint64, name string) [16]byte {
	h := md5.New()
	h.Write(hash16k[:])
	_ = binary.Write(h, binary.LittleEndian, fileSize)
	h.Write([]byte(name))
	var id [16]byte
	copy(id[:], h.Sum(nil))
	return id
}

// MainPacket builds the body of a PAR2 Main packet.
//
// Layout:
//   - [8]  uint64 sliceSize
//   - [4]  uint32 number of recoverable files
//   - [16*N] file IDs of recoverable files (N = len(fileIDs))
func MainPacket(sliceSize uint64, fileIDs [][16]byte) []byte {
	bodySize := 8 + 4 + 16*len(fileIDs)
	body := make([]byte, bodySize)

	binary.LittleEndian.PutUint64(body[0:8], sliceSize)
	binary.LittleEndian.PutUint32(body[8:12], uint32(len(fileIDs))) // number of recoverable files

	offset := 12
	for _, fid := range fileIDs {
		copy(body[offset:offset+16], fid[:])
		offset += 16
	}

	return body
}

// RecoverySetID computes the Recovery Set ID from a Main packet body.
// It is the MD5 of the body of the Main packet.
func RecoverySetID(mainBody []byte) [16]byte {
	return md5.Sum(mainBody)
}

// FileDescriptionPacket builds the body of a File Description packet.
//
// Layout:
//   - [16] fileID
//   - [16] hashFull (MD5 of entire file)
//   - [16] hash16k  (MD5 of first 16KB)
//   - [8]  uint64 fileSize
//   - [N]  filename (null-padded to 4-byte boundary)
func FileDescriptionPacket(fileID [16]byte, hashFull [16]byte, hash16k [16]byte, fileSize uint64, name string) []byte {
	nameBytes := []byte(name)
	// Pad name to 4-byte boundary with nulls
	paddedLen := (len(nameBytes) + 3) &^ 3
	paddedName := make([]byte, paddedLen)
	copy(paddedName, nameBytes)

	bodySize := 16 + 16 + 16 + 8 + paddedLen
	body := make([]byte, bodySize)

	copy(body[0:16], fileID[:])
	copy(body[16:32], hashFull[:])
	copy(body[32:48], hash16k[:])
	binary.LittleEndian.PutUint64(body[48:56], fileSize)
	copy(body[56:], paddedName)

	return body
}

// IFSCEntry represents one entry in an IFSC (Input File Slice Checksum) packet.
type IFSCEntry struct {
	MD5  [16]byte
	CRC32 uint32
}

// IFSCPacket builds the body of an IFSC packet.
//
// Layout:
//   - [16] fileID
//   - For each slice:
//     - [16] MD5 hash of slice
//     - [4]  CRC32 of slice
func IFSCPacket(fileID [16]byte, entries []IFSCEntry) []byte {
	bodySize := 16 + len(entries)*20
	body := make([]byte, bodySize)

	copy(body[0:16], fileID[:])
	offset := 16
	for _, e := range entries {
		copy(body[offset:offset+16], e.MD5[:])
		binary.LittleEndian.PutUint32(body[offset+16:offset+20], e.CRC32)
		offset += 20
	}

	return body
}

// RecoverySlicePacket builds the body of a Recovery Slice packet.
//
// Layout:
//   - [4]  uint32 exponent
//   - [N]  recovery data (sliceSize bytes)
func RecoverySlicePacket(exponent uint16, data []byte) []byte {
	body := make([]byte, 4+len(data))
	binary.LittleEndian.PutUint32(body[0:4], uint32(exponent))
	copy(body[4:], data)
	return body
}

// CreatorPacket builds the body of a Creator packet.
//
// Layout:
//   - [N] creator string (ASCII, null-padded to 4-byte boundary)
func CreatorPacket(creator string) []byte {
	nameBytes := []byte(creator)
	paddedLen := (len(nameBytes) + 3) &^ 3
	body := make([]byte, paddedLen)
	copy(body, nameBytes)
	return body
}

// ComputeSliceChecksums computes MD5 and CRC32 for each slice of a file.
func ComputeSliceChecksums(data []byte, sliceSize int) []IFSCEntry {
	numSlices := (len(data) + sliceSize - 1) / sliceSize
	entries := make([]IFSCEntry, numSlices)

	for i := range numSlices {
		start := i * sliceSize
		end := start + sliceSize
		if end > len(data) {
			end = len(data)
		}
		slice := data[start:end]

		// If this is the last slice and it's shorter than sliceSize, pad with zeros
		if len(slice) < sliceSize {
			padded := make([]byte, sliceSize)
			copy(padded, slice)
			slice = padded
		}

		entries[i].MD5 = md5.Sum(slice)
		entries[i].CRC32 = crc32.ChecksumIEEE(slice)
	}

	return entries
}
