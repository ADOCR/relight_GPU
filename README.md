# RelightLab GPU fork

An experimental, independently built fork of [CNR-ISTI VCLab RelightLab](https://github.com/cnr-isti-vclab/relight), based on **RelightLab 2026.09** (`d65e6f2c367bedb4716366a510f4cd7d9f7f48c1`), integrated with upstream `c9709ddcbd07026e4022c24c5988e5a655b583ed`. [EspaÃ±ol](README.es.md).

The fork preserves the upstream CPU paths, all input images and their original resolution. It does not replace calibrated light directions, alignment or image-quality review.

## Changes

- Optional **strict FP32 CUDA/cuBLAS projection** for directional-light PTM/HSH RGB exports. The RGB block structure is validated before compacting the matrices; transfers use bounded batches and pinned host buffers when available.
- An **exact CPU RGB path** skips only the known zero cross-channel terms. Tests compare it exactly with the original dense FP32 accumulation.
- Optional **FP64 cuFFT** for the existing FFT normal-to-height integration, retaining its normalization and a CPU fallback. Plans and buffers are reused within a thread, with a device budget of 1.5 GiB or half of free VRAM, whichever is smaller.
- Automatic selection keeps small exports and initialization quantiles on CPU. CUDA can be requested explicitly; unavailable hardware, libraries or memory fall back to CPU.
- Windows JPEG mappings remove the CRT open-stream limit for JPEG inputs; the existing limit guard remains for other formats. LP imports accept more than 1,000 directions.
- Recorded `FocalLengthIn35mmFilm` takes priority over physical focal length. Existing EXIF/ICC parsing and color transforms are retained.
- Stage profiling, reproducible synthetic fixtures, accuracy tests and published measurements.

## Use

See the [upstream usage guide](README.upstream.md) for the complete CLI and format documentation.

In **File > Preferences > Performances**, select Automatic, CPU or NVIDIA GPU for PTM/HSH processing. The application has a separate settings namespace, `RelightLab GPU`, so it does not overwrite the normal installation's preferences.

```text
relight-cli -b ptm -p 18 -w 4 -g auto INPUT OUTPUT
relight-cli -b hsh -p 27 -w 4 -g cuda INPUT OUTPUT
```

`-g` accepts `auto`, `cpu` or `cuda`. Unsupported color spaces, positional-light fits and rank-deficient SVD fits retain the original CPU implementation. `RELIGHT_COMPUTE_BACKEND` can select the projection backend, `RELIGHT_FFT_BACKEND` selects `auto|cpu|cuda` for surface FFT, and `RELIGHT_PROFILE=1` prints stage measurements.

Automatic projection selection is deliberately conservative: it requires at least 4,096 RGB components, an export of at least one megapixel and a sufficiently large row operation. Initialization quantiles stay on CPU in automatic mode. Automatic FFT requires at least 4,194,304 complex elements, including the integration padding. These are heuristics, not a guarantee of a speedup.

## Measured results and limits

See [benchmarks](docs/BENCHMARKS.md). Faster GPU kernels do **not** imply an equally faster export. On the tested RTX 3070, loading, decoding, color conversion and writing dominate many small synthetic workloads. The improved CPU path can outperform forced CUDA. No universal speedup is claimed.

nvJPEG was evaluated but was not integrated as the default reader: the tested decode was slower and its RGB output was not pixel-identical to libjpeg. General ICC conversion remains on CPU, preserving the existing transform instead of replacing it with an approximate gamma conversion. Sparse BNI and robust normal estimation remain on CPU; the GPU surface path applies specifically to the existing FFT integration.

## Build and tests

Requirements: C++17, CMake/Ninja, Qt 6 Core/Gui/Widgets/Concurrent/Xml/Svg, JPEG, TIFF, PNG, LittleCMS and Exiv2. CUDA Toolkit headers are optional; runtime CUDA libraries are loaded dynamically. Without CUDA headers, the build remains CPU-capable. No CUDA DLLs are required merely to launch the application.

On Windows, use `scripts/build-windows.ps1` with `-QtRoot`, `-VcpkgRoot` and optionally `-CudaRoot`. Keep the original upstream licenses and third-party licenses when redistributing builds. Benchmark dependencies are in `benchmarks/requirements.txt`.

```text
cmake -S tests/gpu -B build-tests -DCMAKE_PREFIX_PATH=QT_PATH
cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
python benchmarks/generate_fixture.py benchmarks/data/example --images 3138 --width 384 --height 256
python benchmarks/run_export_benchmark.py RELIGHT_CLI benchmarks/data/example benchmarks/outputs/example --result benchmarks/results/example.json
```

The generated data are synthetic. Private photographs, geographic metadata, local planning files, build directories, downloaded tools and runtime binaries are excluded from the shareable source.

## Attribution and licensing

RelightLab is developed by CNR-ISTI VCLab. This fork retains the upstream copyright and license files. Core and GPL-linked components retain their original terms; Exiv2-enabled builds include GPL components. This is an experimental fork, not an official CNR release. See `LICENSE`, `LICENSE-GPL-3.0` and `LICENSE-LGPL-3.0`.
