# Static arithmetic coding vs NNCP

Information and Coding Theory, HW1 Exercise 9 — Sergey Gorelov.

## Submission

- `src/`: C++17 static arithmetic encoder/decoder and experiment driver.
- `third_party/nncp/`: official NNCP v3.3 C sources and the CPU LibNC dependency, with the original copyright and license notices. NNCP is Fabrice Bellard's implementation.
- `datasets/`: the three created files, each 32,768 bytes. Original is the first 32 KiB of Tiny Shakespeare; Shuffled permutes its bytes with seed 42; Repeated repeats its first 128 bytes 256 times.
- `results.json`: saved measurements for both methods.
- `CONCLUSIONS.pdf`: measurements, empirical entropies and conclusions.

The saved experiment used Linux x86-64 through Rosetta on Apple Silicon, NNCP's `default` Transformer profile, CPU, one thread. There is one timing per method/dataset; all six recorded round-trip checks passed. Archive sizes include headers.

## Build and run

The arithmetic coder builds with Clang or GCC on macOS and Linux:

```sh
make
make run RUNS=1
```

To measure both methods on Linux x86-64 with AVX2/FMA, GCC, make and zlib development headers:

```sh
make compare RUNS=1
```

New datasets, compressed files and `results.json` are written under `build/run-new/`. The submitted datasets and measurements stay unchanged. Use `RUNS=5` for five timing samples per pair. The experiment accepts either a directory of the three prepared datasets or a Tiny Shakespeare text file; see `./build/hw1_ex9 --help`.

## Sources

Dataset: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

NNCP: https://bellard.org/nncp/ — v3.3, release 2024-06-05, MIT source license. The upstream CPU LibNC library is included because it is required to link and run the official compressor; upstream documentation permits redistribution.
