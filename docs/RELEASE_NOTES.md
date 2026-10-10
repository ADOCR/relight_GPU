# Experimental GPU fork release notes

Based on upstream RelightLab 2026.09. Adds optional strict FP32 PTM/HSH RGB projection, exact channel-separated CPU projection, optional FP64 FFT integration, bounded buffers, CPU fallback and conservative automatic selection. Windows JPEG stream-limit and EXIF focal improvements are included.

Integrated with upstream `c9709ddcbd07026e4022c24c5988e5a655b583ed`: keeps float input/color processing, parallel image rows, parallel quantiles and JPEG encoding, batched RBF/bilinear projection and streamlined CPU surface integration. Optional GPU projection reads the original row when resampling is unnecessary; cuFFT works on the streamlined buffers with the upstream normalization and multi-threaded CPU fallback. CLI accepts both `-g` and the upstream `-H` option.

No input-image reduction or downscaling is performed. Source EXIF/ICC parsing and LittleCMS transforms are retained. Unsupported processing modes keep CPU. nvJPEG, BNI GPU solving and generic GPU ICC conversion are not enabled; measured limitations are documented.

This fork is intended for review and reproducible testing. It is not an official CNR release. Compare project outputs and validate light calibration before treating any RTI or inferred surface as analytical evidence.
