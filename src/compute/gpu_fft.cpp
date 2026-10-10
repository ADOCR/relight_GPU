#include "gpu_fft.h"
#include <QCoreApplication>
#include <QLibrary>
#include <QStringList>
#include <QSettings>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifndef RELIGHT_HAS_CUDA
#define RELIGHT_HAS_CUDA 0
#endif
#if RELIGHT_HAS_CUDA
#include <cuda_runtime_api.h>
#include <cufft.h>
#endif
namespace relight {
struct GpuFft2D::Impl {
    size_t budget,width=0,height=0,bytes=0,allocated=0;
    bool ready=false;
    std::string message="CUDA FFT not initialized";
    std::vector<std::complex<double>> result;
#if RELIGHT_HAS_CUDA
    QLibrary runtime,library;
    cufftHandle plan=0;
    void *device=nullptr,*workspace=nullptr;
#define API(name) decltype(&name) p_##name=nullptr
    API(cudaGetDeviceCount);API(cudaMemGetInfo);API(cudaMalloc);API(cudaFree);API(cudaMemcpy);
    API(cufftCreate);API(cufftDestroy);API(cufftSetAutoAllocation);API(cufftMakePlan2d);
    API(cufftSetWorkArea);API(cufftExecZ2Z);
#undef API
    bool load(QLibrary& lib,const QString& name){
        const QStringList paths{QCoreApplication::applicationDirPath()+"/"+name,name,
            qEnvironmentVariable("CUDA_PATH")+"/bin/x64/"+name,qEnvironmentVariable("CUDA_PATH")+"/bin/"+name};
        for(const auto&path:paths){lib.setFileName(path);if(lib.load())return true;}
        message="CUDA FFT library unavailable: "+name.toStdString();return false;
    }
    bool libraries(){
#ifdef _WIN32
        const QString runtimeName="cudart64_"+QString::number(CUDART_VERSION/1000);
        const QString fftName="cufft64_"+QString::number(CUFFT_VER_MAJOR);
#else
        const QString runtimeName="libcudart.so."+QString::number(CUDART_VERSION/1000);
        const QString fftName="libcufft.so."+QString::number(CUFFT_VER_MAJOR);
#endif
        if(!load(runtime,runtimeName)||!load(library,fftName))return false;
#define GET(lib,name) p_##name=reinterpret_cast<decltype(p_##name)>(lib.resolve(#name));if(!p_##name){message="Missing FFT symbol: " #name;return false;}
        GET(runtime,cudaGetDeviceCount);GET(runtime,cudaMemGetInfo);GET(runtime,cudaMalloc);GET(runtime,cudaFree);GET(runtime,cudaMemcpy);
        GET(library,cufftCreate);GET(library,cufftDestroy);GET(library,cufftSetAutoAllocation);GET(library,cufftMakePlan2d);
        GET(library,cufftSetWorkArea);GET(library,cufftExecZ2Z);
#undef GET
        return true;
    }
    void release(){
        if(plan&&p_cufftDestroy)p_cufftDestroy(plan);plan=0;
        if(device&&p_cudaFree)p_cudaFree(device);device=nullptr;
        if(workspace&&p_cudaFree)p_cudaFree(workspace);workspace=nullptr;
        ready=false;allocated=0;
    }
#else
    void release(){}
#endif
    explicit Impl(size_t limit):budget(limit){}
    ~Impl(){release();}
};
GpuFft2D::GpuFft2D(size_t budget):impl(new Impl(budget)){}
GpuFft2D::~GpuFft2D()=default;
const std::string& GpuFft2D::diagnostic()const{return impl->message;}
size_t GpuFft2D::deviceBytes()const{return impl->allocated;}
bool GpuFft2D::configure(size_t width,size_t height){
    auto&s=*impl;
    if(s.ready&&s.width==width&&s.height==height)return true;
    s.release();
#if RELIGHT_HAS_CUDA
    if(!width||!height||width>size_t(std::numeric_limits<int>::max())||height>size_t(std::numeric_limits<int>::max())
        ||width>std::numeric_limits<size_t>::max()/height/sizeof(std::complex<double>))return false;
    s.bytes=width*height*sizeof(std::complex<double>);
    if(s.bytes>s.budget){s.message="FFT input exceeds GPU memory budget";return false;}
    if(!s.libraries())return false;
    int devices=0;size_t freeBytes=0,totalBytes=0;
    if(s.p_cudaGetDeviceCount(&devices)!=cudaSuccess||devices<1||s.p_cudaMemGetInfo(&freeBytes,&totalBytes)!=cudaSuccess){s.message="CUDA FFT device unavailable";return false;}
    size_t workBytes=0;
    if(s.p_cufftCreate(&s.plan)!=CUFFT_SUCCESS||s.p_cufftSetAutoAllocation(s.plan,0)!=CUFFT_SUCCESS
        ||s.p_cufftMakePlan2d(s.plan,int(height),int(width),CUFFT_Z2Z,&workBytes)!=CUFFT_SUCCESS){s.message="CUDA FFT plan failed";s.release();return false;}
    const size_t limit=std::min(s.budget,freeBytes/2);
    if(s.bytes>limit||workBytes>limit-s.bytes){s.message="CUDA FFT workspace exceeds budget";s.release();return false;}
    if(s.p_cudaMalloc(&s.device,s.bytes)!=cudaSuccess
        ||(workBytes&&s.p_cudaMalloc(&s.workspace,workBytes)!=cudaSuccess)
        ||s.p_cufftSetWorkArea(s.plan,s.workspace)!=CUFFT_SUCCESS){s.message="CUDA FFT allocation failed";s.release();return false;}
    s.result.resize(width*height);s.width=width;s.height=height;s.allocated=s.bytes+workBytes;s.ready=true;
    s.message="CUDA FFT double precision ready";return true;
#else
    (void)width;(void)height;s.message="Built without CUDA FFT; using CPU";return false;
#endif
}
bool GpuFft2D::transform(std::complex<double>*data,bool inverse,double normalization){
    auto&s=*impl;
#if RELIGHT_HAS_CUDA
    if(!s.ready||!data||!std::isfinite(normalization))return false;
    if(s.p_cudaMemcpy(s.device,data,s.bytes,cudaMemcpyHostToDevice)!=cudaSuccess
        ||s.p_cufftExecZ2Z(s.plan,static_cast<cufftDoubleComplex*>(s.device),static_cast<cufftDoubleComplex*>(s.device),inverse?CUFFT_INVERSE:CUFFT_FORWARD)!=CUFFT_SUCCESS
        ||s.p_cudaMemcpy(s.result.data(),s.device,s.bytes,cudaMemcpyDeviceToHost)!=cudaSuccess){s.message="CUDA FFT execution failed";s.release();return false;}
    // Only replace caller data after the complete transform has succeeded.
    for(auto&v:s.result){v*=normalization;if(!std::isfinite(v.real())||!std::isfinite(v.imag())){s.message="Non-finite FFT result";return false;}}
    std::copy(s.result.begin(),s.result.end(),data);return true;
#else
    (void)data;(void)inverse;(void)normalization;return false;
#endif
}
bool tryGpuFft2D(std::complex<double>*data,size_t width,size_t height,bool inverse,double normalization){
    QString policy=qEnvironmentVariable("RELIGHT_FFT_BACKEND").toLower();
    if(policy.isEmpty())policy=QSettings().value("ComputeBackend","auto").toString();
    if(policy=="cpu"||(policy=="auto"&&double(width)*height<4194304.0))return false;
    thread_local GpuFft2D gpu;
    try{return gpu.configure(width,height)&&gpu.transform(data,inverse,normalization);}
    catch(...){return false;}
}
}
