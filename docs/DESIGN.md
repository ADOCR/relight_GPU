# Design and limitations

Projection uses the original material projection matrices. GPU computation performs FP32 GEMM in pedantic mode, without TF32/FP16 reduction. Separate RGB blocks are validated exactly before compacting; other matrices use the full layout. Every image contributes and no resolution reduction is performed.

Each projection worker requests at most 32 MiB for matrix/input/output device buffers and uses small batches. CUDA contexts and library workspaces have additional overhead. Free-memory checks and allocation failures preserve the CPU fallback. Host transfer buffers can be pinned; if pinning fails, ordinary host memory is used.

FFT uses double-precision complex transforms. Input and workspace allocations are checked against 1.5 GiB and half of available VRAM. Caller data are only replaced after a complete, finite result is downloaded. Existing integration normalization and padding are retained. The FFT plan/buffers are reused for matching dimensions within a thread.

JPEG decoding, EXIF/ICC parsing and LittleCMS conversion remain the original paths. nvJPEG evaluation does not justify enabling it on the tested GPU: decode was slower and returned RGB differences. A future fully streamed GPU pipeline would need separate decoding, ICC and geometric validation; it is not part of the current implementation.

GPU acceleration currently applies to directional-light PTM/HSH RGB projection and the existing complex FFT integration of normals. Positional lights, other color models, SVD fallback, BNI and robust normal fitting retain CPU. Automatic thresholds are conservative heuristics, and the measured end-to-end benefit depends heavily on decoding, color conversion and disk/cache behavior.

For Spanish MSVC installations where CMake misdecodes the /showIncludes prefix, configure `RELIGHT_MSVC_DEPS_PREFIX` or pass `-MsvcDepsPrefix 'Nota: inclusión del archivo: '` to the Windows build script. This makes Ninja track header dependencies correctly. Keep /EHsc and /GR when adding /utf-8.
