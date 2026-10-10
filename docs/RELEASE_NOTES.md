# Experimental GPU fork release notes

Based on upstream RelightLab 2026.09. Adds optional strict FP32 PTM/HSH RGB projection, exact channel-separated CPU projection, optional FP64 FFT integration, bounded buffers, CPU fallback and conservative automatic selection. Windows JPEG stream-limit and EXIF focal improvements are included.

No input-image reduction or downscaling is performed. Source EXIF/ICC parsing and LittleCMS transforms are retained. Unsupported processing modes keep CPU. nvJPEG, BNI GPU solving and generic GPU ICC conversion are not enabled; measured limitations are documented.

This fork is intended for review and reproducible testing. It is not an official CNR release. Compare project outputs and validate light calibration before treating any RTI or inferred surface as analytical evidence.
