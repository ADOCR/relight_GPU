# RelightLab con mejoras CPU y GPU

Versión experimental e independiente basada en RelightLab 2026.09 de CNR-ISTI VCLab. [Documentación técnica en inglés](README.md).

## Qué mejora

- Cálculos PTM/HSH RGB por bloques con NVIDIA GPU, con precisión FP32 normal y alternativa CPU.
- Cálculo CPU más eficiente que conserva exactamente la acumulación original por canal.
- Transformadas de Fourier en doble precisión para la integración de normales a alturas.
- Lectura de todos los JPG en Windows sin el límite de 512/8.192 flujos de archivos de la biblioteca C.
- Importación de más de 1.000 direcciones de luz y uso correcto de la focal equivalente registrada en EXIF.

Las fotos conservan su resolución y los originales permanecen intactos. Las luces, la alineación y la calibración deben ser válidas; estas mejoras no las sustituyen.

## Cómo usarla

Abrir la versión independiente **RelightLab GPU**. En **File > Preferences > Performances**, elegir **Automatic**, **CPU** o **NVIDIA GPU** para PTM/HSH. Automatic conserva CPU en trabajos pequeños y usa GPU en los casos previstos por sus criterios. Si faltan bibliotecas, una tarjeta compatible o memoria suficiente, continúa con CPU.

No todos los modos tienen GPU: colores distintos de RGB, luces posicionales y ajustes que necesitan SVD mantienen la implementación CPU. La mejora de superficie afecta a la integración FFT existente, no al estimador robusto ni al solver BNI.

## Resultados reales

Las mediciones y sus límites están en [BENCHMARKS.md](docs/BENCHMARKS.md). Se comparó el tiempo completo de exportación además del cálculo aislado. En las pruebas pequeñas, la CPU optimizada puede ser más rápida que forzar GPU. Por eso no se promete una aceleración universal.

Se evaluó nvJPEG y se mantuvo el lector original: en la RTX 3070 probada fue más lento y produjo diferencias de píxeles. La conversión ICC continúa con el motor original de color.

## Carpeta para compartir

Incluye código fuente, pruebas, generadores de datos sintéticos, mediciones y scripts de compilación. Los datos privados, GPS, vídeos, fotos reales, herramientas descargadas y binarios locales quedan fuera del código compartible. El repositorio no se ha publicado automáticamente.

Conservar la atribución a CNR-ISTI VCLab y las licencias originales. Esta versión es un fork experimental, no una publicación oficial de CNR.
