# Measurements and numerical validation

Test system: NVIDIA GeForce RTX 3070 (8 GiB), AMD Ryzen 5 4600G (6 cores / 12 threads), 32 GiB RAM, Windows, MSVC, CUDA Toolkit 13.4. All publishable inputs are synthetic. JPEG fixtures use quality 98, 4:4:4 and an embedded sRGB profile.

## Results

The timing table below was measured on the original 2026.09-based fork, **before** integrating upstream commit `c9709ddcbd07026e4022c24c5988e5a655b583ed`. That integration adds parallel image reading, float color transforms, parallel quantiles/encoding and a streamlined multi-threaded CPU surface implementation. These historical speed ratios must not be presented as measurements of the integrated version. Integration validation checks correctness; new performance claims require fresh end-to-end benchmarks on an idle machine.

| Measurement | CPU | GPU | Interpretation |
|---|---:|---:|---|
| Dense projection: 9,414 components, 18 planes, 1,024 pixels | 142.993 ms | 18.816 ms | GPU time includes host/device transfers; excludes image loading and writing. This reference predates the exact CPU channel optimization. |
| Initial complete PTM export: 3,138 images, 128 x 96, dense CPU reference | 4.432 s median | 3.740 s median | About 1.185x. The first cold CPU load took 24.27 s; it must not be reported as a GPU speedup. |
| Complete PTM: 3,138 images, 384 x 256, optimized CPU vs compact pinned GPU | 14.647 s median | 14.992 s median | GPU was slightly slower (0.977x). Automatic mode retains CPU for this sub-megapixel workload. |
| Complete normal-to-height FFT integration: synthetic 1,024 x 1,024 normals, padded internally | 1,978.533 ms | 1,442.770 ms | About 1.371x including integration and padding; final float height arrays were identical in this test. |
| FP64 FFT: 2,048 x 2,048 complex elements | 246.739 ms | 36.764 ms | GPU time includes transfers. Plan/startup was 129.828 ms, so plans are reused and small jobs remain on CPU. |
| Decode 64 synthetic 384 x 256 JPEGs, CPU with four threads | 8.803 ms | nvJPEG 34.674 ms | nvJPEG was not adopted. It had no dedicated JPEG decode engines on this card and changed some RGB values. |

The complete-export runner alternates CPU/GPU order and records every wall-clock sample and per-stage profile. The OS file cache is not purged. Medians are not a substitute for measuring a real project's workload; the underlying JSON records expose cold/warm variation. Current measurements do not establish a speedup for a full-resolution real archaeological RTI project.

## Accuracy

- The exact CPU channel projection matched the original dense FP32 reference exactly for 1, 7, 128, 3,138 and 20,000 lights with 3/9/18/27 planes.
- Strict FP32 GPU RGB tests covered 7, 3,138 and 20,000 lights, multiple batch boundaries and 9/18/27 planes. Maximum observed difference against CPU was below 0.0000124 in those tests.
- Complete PTM export comparisons evaluate reconstructed intensities over 12 light directions. The 3,138-image 384 x 256 test had maximum difference 1.287 on a 0-255 scale and mean difference 0.00370 after coefficient quantization and JPEG encoding. Normalized coefficient JPEGs are not necessarily byte-identical.
- HSH comparisons use the upstream HSH basis and the same reconstructed-intensity check. See the JSON result files for each fixture.
- Output ICC transforms are required to match exactly. The test ignores only ICC creation date and profile-ID header fields; it records whether complete ICC bytes match. Different run timestamps are not a color change.
- FP64 FFTs matched pocketfft to within 0.000000000004 on the tested dimensions, and inverse transforms passed round-trip checks. The full surface-integration test produced identical final float heights.
- nvJPEG decoding differed by up to three RGB levels, with mean difference 0.0213, so its performance result does not justify replacing the current reader.

## Reproduce

Use `benchmarks/generate_fixture.py` and `benchmarks/run_export_benchmark.py`. Results are in `benchmarks/results/`; generated input and output directories are excluded from Git. Use `tests/gpu` for projection, CPU fallback, RGB matrix validation, memory limits, FFT precision and surface consistency.

Set `RELIGHT_REQUIRE_CUDA=1` when testing on a GPU host so unavailable CUDA is a failure. On CPU-only CI, CUDA tests are skipped with return code 77, and CPU fallback / exact CPU tests still run. `nvjpeg_benchmark` is an optional diagnostic tool and does not change the application's JPEG reader.

## Validation after upstream integration

The integrated Windows application compiled successfully and all six CPU/CUDA tests passed with `RELIGHT_REQUIRE_CUDA=1` on the RTX 3070. `tests/integration/verify_export_merge.py` additionally validates eight complete exports from 512 synthetic input images: PTM/HSH with actual CUDA calls, and RBF/bilinear with the CPU fallback and different worker counts. It checks image dimensions, coefficient planes, all 512 light directions, finite materials and matching ICC transforms. PTM exercises `-H` and an offset crop; odd crop widths exercise partial four-pixel RBF batches.

The maximum reconstructed CPU/CUDA difference was 0.201/255 for cropped PTM and 0.354/255 for HSH. RBF and bilinear coefficient images were identical between one and two row workers. See `benchmarks/results/merge_validation.json`. These checks establish numerical consistency on the synthetic fixture, not a new speedup or validation of real calibrated geometry.

```text
python tests/integration/verify_export_merge.py RELIGHT_CLI SYNTHETIC_FIXTURE NEW_OUTPUT_DIRECTORY
```
