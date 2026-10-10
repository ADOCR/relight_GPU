#include "gpu_projection.h"
#include <QCoreApplication>
#include <QLibrary>
#include <QStringList>
#include <algorithm>
#include <cmath>
#include <limits>

#ifndef RELIGHT_HAS_CUDA
#define RELIGHT_HAS_CUDA 0
#endif
#if RELIGHT_HAS_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

namespace relight {
struct GpuProjection::Impl {
    size_t budget, components = 0, planes = 0, capacity = 0, allocated = 0;
    const float* matrixIdentity = nullptr;
    bool attempted = false, ready = false, channelSeparated = false;
    std::string message = "CUDA backend not initialized";
    std::vector<float> input, output;
#if RELIGHT_HAS_CUDA
    QLibrary runtime, blas;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    float *matrix = nullptr, *values = nullptr, *result = nullptr;
    float *pinnedInput = nullptr, *pinnedOutput = nullptr;
#define CUDA_API(name) decltype(&name) p_##name = nullptr
    CUDA_API(cudaGetDeviceCount);
    CUDA_API(cudaMemGetInfo);
    CUDA_API(cudaMalloc);
    CUDA_API(cudaFree);
    CUDA_API(cudaHostAlloc);
    CUDA_API(cudaFreeHost);
    CUDA_API(cudaGetLastError);
    CUDA_API(cudaMemcpyAsync);
    CUDA_API(cudaStreamCreateWithFlags);
    CUDA_API(cudaStreamSynchronize);
    CUDA_API(cudaStreamDestroy);
    CUDA_API(cudaGetErrorString);
    CUDA_API(cublasCreate_v2);
    CUDA_API(cublasDestroy_v2);
    CUDA_API(cublasSetStream_v2);
    CUDA_API(cublasSetMathMode);
    CUDA_API(cublasSgemm_v2);
#undef CUDA_API
    bool load(QLibrary& library, const QString& name) {
        QStringList candidates{QCoreApplication::applicationDirPath() + "/" + name,
            name, qEnvironmentVariable("CUDA_PATH") + "/bin/x64/" + name,
            qEnvironmentVariable("CUDA_PATH") + "/bin/" + name};
        for (const auto& candidate : candidates) {
            library.setFileName(candidate);
            if (library.load()) return true;
        }
        message = "CUDA library unavailable: " + name.toStdString();
        return false;
    }
    bool libraries() {
        const int major = CUDART_VERSION / 1000;
        const QString suffix = QString::number(major);
#ifdef _WIN32
        const QString runtimeName = "cudart64_" + suffix;
        const QString blasName = "cublas64_" + suffix;
#else
        const QString runtimeName = "libcudart.so." + suffix;
        const QString blasName = "libcublas.so." + suffix;
#endif
        if (!load(runtime, runtimeName) || !load(blas, blasName)) return false;
#define RESOLVE(lib, name) p_##name = reinterpret_cast<decltype(p_##name)>(lib.resolve(#name)); if (!p_##name) { message = "Missing CUDA symbol: " #name; return false; }
        RESOLVE(runtime, cudaGetDeviceCount);
        RESOLVE(runtime, cudaMemGetInfo);
        RESOLVE(runtime, cudaMalloc);
        RESOLVE(runtime, cudaFree);
        RESOLVE(runtime, cudaHostAlloc);
        RESOLVE(runtime, cudaFreeHost);
        RESOLVE(runtime, cudaGetLastError);
        RESOLVE(runtime, cudaMemcpyAsync);
        RESOLVE(runtime, cudaStreamCreateWithFlags);
        RESOLVE(runtime, cudaStreamSynchronize);
        RESOLVE(runtime, cudaStreamDestroy);
        RESOLVE(runtime, cudaGetErrorString);
        RESOLVE(blas, cublasCreate_v2);
        RESOLVE(blas, cublasDestroy_v2);
        RESOLVE(blas, cublasSetStream_v2);
        RESOLVE(blas, cublasSetMathMode);
        RESOLVE(blas, cublasSgemm_v2);
#undef RESOLVE
        return true;
    }
    bool cudaOk(cudaError_t code, const char* stage) {
        if (code == cudaSuccess) return true;
        message = std::string(stage) + ": " + p_cudaGetErrorString(code);
        ready = false;
        return false;
    }
    bool blasOk(cublasStatus_t code, const char* stage) {
        if (code == CUBLAS_STATUS_SUCCESS) return true;
        message = std::string(stage) + ": cuBLAS status " + std::to_string(int(code));
        ready = false;
        return false;
    }
    void release() {
        if (stream && p_cudaStreamSynchronize) p_cudaStreamSynchronize(stream);
        if (handle && p_cublasDestroy_v2) p_cublasDestroy_v2(handle);
        handle = nullptr;
        if (matrix && p_cudaFree) p_cudaFree(matrix);
        if (values && p_cudaFree) p_cudaFree(values);
        if (result && p_cudaFree) p_cudaFree(result);
        matrix = values = result = nullptr;
        if (pinnedInput && p_cudaFreeHost) p_cudaFreeHost(pinnedInput);
        if (pinnedOutput && p_cudaFreeHost) p_cudaFreeHost(pinnedOutput);
        pinnedInput = pinnedOutput = nullptr;
        if (stream && p_cudaStreamDestroy) p_cudaStreamDestroy(stream);
        stream = nullptr;
        ready = false;
        allocated = 0;
    }
#else
    void release() {}
#endif
    explicit Impl(size_t limit) : budget(limit) {}
    ~Impl() { release(); }
};

GpuProjection::GpuProjection(size_t budget) : impl(new Impl(budget)) {}
GpuProjection::~GpuProjection() = default;
bool GpuProjection::compiledWithCuda() { return RELIGHT_HAS_CUDA != 0; }
const std::string& GpuProjection::diagnostic() const { return impl->message; }
size_t GpuProjection::batchCapacity() const { return impl->capacity; }
size_t GpuProjection::deviceBytes() const { return impl->allocated; }

bool GpuProjection::configure(const float* projection, size_t k, size_t m, bool channelSeparated) {
    auto& s = *impl;
    if (s.attempted) return s.ready && s.matrixIdentity == projection && s.components == k && s.planes == m && s.channelSeparated == channelSeparated;
    s.attempted = true;
#if RELIGHT_HAS_CUDA
    if (!projection || !k || !m || k > size_t(std::numeric_limits<int>::max()) || m > 1024
        || k > std::numeric_limits<size_t>::max() / (m * sizeof(float))
        || (channelSeparated && (k%3 || m%3))) {
        s.message = "Invalid projection dimensions"; return false;
    }
    const size_t matrixBytes = (channelSeparated ? (k/3)*(m/3) : k*m) * sizeof(float);
    if (matrixBytes >= s.budget) { s.message = "Projection exceeds GPU memory budget"; return false; }
    for (size_t i = 0; i < k*m; ++i) if (!std::isfinite(projection[i])) {
        s.message = "Non-finite projection matrix"; return false;
    }
    std::vector<float> compact;
    const float* matrixData = projection;
    if (channelSeparated) {
        compact.resize((k/3)*(m/3));
        for (size_t p=0;p<m/3;++p) for (size_t light=0;light<k/3;++light)
            compact[p*(k/3)+light]=projection[p*3*k+light*3];
        // Validate the promised block structure before omitting any terms.
        for (size_t p=0;p<m;++p) for (size_t i=0;i<k;++i) {
            const float expected=(p%3==i%3)?compact[(p/3)*(k/3)+i/3]:0.0f;
            if (projection[p*k+i]!=expected) {s.message="Projection is not channel-separated RGB";return false;}
        }
        matrixData=compact.data();
    }
    if (!s.libraries()) return false;
    int count = 0;
    if (!s.cudaOk(s.p_cudaGetDeviceCount(&count), "CUDA device discovery")) return false;
    if (count == 0) {s.message = "No CUDA device available";return false;}
    size_t freeBytes = 0, totalBytes = 0;
    if (!s.cudaOk(s.p_cudaMemGetInfo(&freeBytes, &totalBytes), "CUDA memory query")) return false;
    const size_t usable = std::min(s.budget, freeBytes / 4);
    const size_t bytesPerPixel = (k + m) * sizeof(float);
    if (usable <= matrixBytes || usable - matrixBytes < bytesPerPixel) {
        s.message = "Insufficient CUDA memory within budget"; return false;
    }
    s.capacity = std::min<size_t>((usable-matrixBytes)/bytesPerPixel, 4096);
    s.components = k; s.planes = m; s.matrixIdentity = projection; s.channelSeparated = channelSeparated;
        if (s.p_cudaHostAlloc(reinterpret_cast<void**>(&s.pinnedInput),k*s.capacity*sizeof(float),cudaHostAllocDefault)!=cudaSuccess) {
        s.pinnedInput=nullptr;s.p_cudaGetLastError();s.input.resize(k*s.capacity);
    }
    if (s.p_cudaHostAlloc(reinterpret_cast<void**>(&s.pinnedOutput),m*s.capacity*sizeof(float),cudaHostAllocDefault)!=cudaSuccess) {
        s.pinnedOutput=nullptr;s.p_cudaGetLastError();s.output.resize(m*s.capacity);
    }
    if (!s.cudaOk(s.p_cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking), "CUDA stream")
        || !s.blasOk(s.p_cublasCreate_v2(&s.handle), "cuBLAS initialization")
        || !s.blasOk(s.p_cublasSetStream_v2(s.handle, s.stream), "cuBLAS stream")
        || !s.blasOk(s.p_cublasSetMathMode(s.handle, CUBLAS_PEDANTIC_MATH), "Strict FP32 math")
        || !s.cudaOk(s.p_cudaMalloc(reinterpret_cast<void**>(&s.matrix), matrixBytes), "Projection allocation")
        || !s.cudaOk(s.p_cudaMalloc(reinterpret_cast<void**>(&s.values), k*s.capacity*sizeof(float)), "Input allocation")
        || !s.cudaOk(s.p_cudaMalloc(reinterpret_cast<void**>(&s.result), m*s.capacity*sizeof(float)), "Output allocation")
        || !s.cudaOk(s.p_cudaMemcpyAsync(s.matrix, matrixData, matrixBytes, cudaMemcpyHostToDevice, s.stream), "Projection upload")
        || !s.cudaOk(s.p_cudaStreamSynchronize(s.stream), "Projection synchronization")) {
        s.release(); return false;
    }
    s.allocated = matrixBytes + (k+m)*s.capacity*sizeof(float);
    s.ready = true; s.message = "CUDA FP32 projection ready (TF32/FP16 disabled)";
    return true;
#else
    (void)projection; (void)k; (void)m;
    s.message = "Built without CUDA headers; using CPU"; return false;
#endif
}

bool GpuProjection::project(const std::vector<const float*>& pixels, const float* mean,
                            std::vector<float>& coefficients) {
    auto& s = *impl;
#if RELIGHT_HAS_CUDA
    if (!s.ready || !mean || pixels.size() > std::numeric_limits<size_t>::max()/s.planes) return false;
    coefficients.resize(pixels.size()*s.planes);
    const float alpha = 1.0f, beta = 0.0f;
    float* hostInput=s.pinnedInput?s.pinnedInput:s.input.data();
    float* hostOutput=s.pinnedOutput?s.pinnedOutput:s.output.data();
    const size_t inputRows=s.channelSeparated?s.components/3:s.components;
    const size_t outputRows=s.channelSeparated?s.planes/3:s.planes;
    for (size_t first = 0; first < pixels.size(); first += s.capacity) {
        const size_t n = std::min(s.capacity, pixels.size()-first);
        for (size_t x = 0; x < n; ++x) {
            if (!pixels[first+x]) return false;
            if (s.channelSeparated) {
                for (size_t light=0;light<inputRows;++light) for (size_t c=0;c<3;++c)
                    hostInput[(x*3+c)*inputRows+light]=pixels[first+x][light*3+c]-mean[light*3+c];
            } else {
                for (size_t k=0;k<s.components;++k)hostInput[x*s.components+k]=pixels[first+x][k]-mean[k];
            }
        }
        if (!s.cudaOk(s.p_cudaMemcpyAsync(s.values, hostInput, n*s.components*sizeof(float), cudaMemcpyHostToDevice, s.stream), "Input upload")
            || !s.blasOk(s.p_cublasSgemm_v2(s.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                int(outputRows), int(s.channelSeparated?n*3:n), int(inputRows), &alpha, s.matrix, int(inputRows),
                s.values, int(inputRows), &beta, s.result, int(outputRows)), "Coefficient projection")
            || !s.cudaOk(s.p_cudaMemcpyAsync(hostOutput, s.result, n*s.planes*sizeof(float), cudaMemcpyDeviceToHost, s.stream), "Result download")
            || !s.cudaOk(s.p_cudaStreamSynchronize(s.stream), "Result synchronization")) {
            s.release(); coefficients.clear(); return false;
        }
        for (size_t i = 0; i < n*s.planes; ++i) {
            const size_t x=i/s.planes,p=i%s.planes;
            const float value=hostOutput[s.channelSeparated?((x*3+p%3)*outputRows+p/3):i];
            if (!std::isfinite(value)) {s.message="Non-finite CUDA result";s.release();coefficients.clear();return false;}
            coefficients[first*s.planes+i]=value;
        }
    }
    return true;
#else
    (void)pixels; (void)mean; (void)coefficients; return false;
#endif
}
}
