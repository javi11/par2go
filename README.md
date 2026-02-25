# par2go

Pure Go library for creating **PAR2** parity/recovery files. PAR2 files let you repair or verify data using Reed–Solomon erasure coding. Recovery files produced by par2go are compatible with [par2cmdline](https://github.com/Parchive/par2cmdline), [MultiPar](https://github.com/Yutaka-Sawada/MultiPar), and other PAR2-compliant tools.

## Installation

```bash
go get github.com/javi11/par2go
```

Requires **Go 1.26** or later.

### CGO-accelerated backend (recommended)

By default par2go links a pre-built [ParPar](https://github.com/animetosho/ParPar) GF(2^16) static library that provides SIMD-optimized Reed-Solomon encoding (SSE2, AVX2, AVX-512, NEON, SVE2, etc. with runtime CPU detection). This requires a C compiler at build time:

| Platform | Requirement |
|----------|-------------|
| macOS | Xcode Command Line Tools (`xcode-select --install`) |
| Linux | GCC or Clang (`apt install build-essential` / `dnf install gcc gcc-c++`) |
| Windows | [MSYS2](https://www.msys2.org/) MinGW-w64 toolchain (`pacman -S mingw-w64-x86_64-gcc`) |

No C++ compiler is needed — the static libraries are pre-built and committed. Only a C linker (the default `cc` / `gcc`) is required by CGo.

### Pure-Go fallback

To build without CGO (no C compiler needed), disable it:

```bash
CGO_ENABLED=0 go build ./...
```

Or use the `purego` build tag:

```bash
go build -tags purego ./...
```

The pure-Go backend uses lookup-table-based GF(2^16) arithmetic. It is fully functional but slower than the SIMD path.

## Quick start

```go
package main

import (
	"context"
	"log"

	"github.com/javi11/par2go"
)

func main() {
	ctx := context.Background()
	outputPath := "/path/to/myfile.par2"
	inputFiles := []string{"/path/to/myfile.bin"}

	opts := par2go.Options{
		SliceSize:   32768,  // 32 KB blocks (must be a multiple of 4)
		NumRecovery: 5,      // number of recovery blocks
	}

	if err := par2go.Create(ctx, outputPath, inputFiles, opts); err != nil {
		log.Fatal(err)
	}
	// Creates myfile.par2 and volume files like myfile.vol00+01.par2, ...
}
```

## Options

| Option          | Description                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `SliceSize`     | Block size in bytes. Must be a positive multiple of 4 (e.g. 32768 for 32 KB).                                                  |
| `NumRecovery`   | Number of recovery blocks to create. You can recover up to this many missing/corrupt blocks.                                   |
| `MemoryBudget`  | Max memory for recovery buffers (default: 512 MB).                                                                             |
| `NumGoroutines` | Parallel workers (default: `runtime.NumCPU()`).                                                                                |
| `OnProgress`    | Optional callback `func(phase string, pct float64)` with phase `"hashing"`, `"encoding"`, or `"writing"` and `pct` in 0.0–1.0. |
| `Creator`       | Creator string stored in the PAR2 file (default: `"Postie"`).                                                                  |

## Progress and cancellation

Use `OnProgress` for UI or logging:

```go
opts := par2go.Options{
	SliceSize:   32768,
	NumRecovery: 5,
	OnProgress: func(phase string, pct float64) {
		fmt.Printf("%s: %.0f%%\n", phase, pct*100)
	},
}
```

Pass a cancellable context to stop creation early:

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()
// call cancel() to abort
err := par2go.Create(ctx, outputPath, inputFiles, opts)
```

## Multiple input files

You can protect several files in one recovery set. All are hashed and encoded together; the main `.par2` and volume files reference the whole set.

```go
inputFiles := []string{
	"/path/to/file1.bin",
	"/path/to/file2.bin",
	"/path/to/file3.bin",
}
err := par2go.Create(ctx, "/path/to/set.par2", inputFiles, opts)
```

## Output files

- **Main file**: the path you pass (e.g. `myfile.par2`). Contains metadata only (Main, Creator, File Description, IFSC packets).
- **Volume files**: created next to the main file with names like `myfile.vol00+01.par2`, `myfile.vol01+01.par2`, `myfile.vol02+02.par2`, … and contain the recovery slice packets.

Use the main `.par2` with any PAR2 repair/verify tool; it will find the volume files by naming convention.

## Compatibility

par2go only **creates** PAR2 files. To verify or repair, use an existing tool (e.g. par2cmdline or MultiPar) that reads the same [PAR2 format](https://github.com/Parchive/par2cmdline). The generated packets and volume layout follow the usual PAR2 conventions.

## License

See [LICENSE](LICENSE).
