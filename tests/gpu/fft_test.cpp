#include "../../src/compute/gpu_fft.h"
#include "../../src/normals/pocketfft.h"
#include <QCoreApplication>
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
#include <vector>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point first){return std::chrono::duration<double,std::milli>(Clock::now()-first).count();}
int main(int argc,char**argv){QCoreApplication app(argc,argv);std::mt19937 random(777);std::uniform_real_distribution<double> value(-1,1);
relight::GpuFft2D availability;if(!availability.configure(16,16)){std::cout<<"SKIP: "<<availability.diagnostic()<<"\n";return qEnvironmentVariableIntValue("RELIGHT_REQUIRE_CUDA")?1:77;}
for(const auto&size:std::vector<std::pair<size_t,size_t>>{{17,13},{128,96},{1024,1024},{2048,2048}}){
size_t w=size.first,h=size.second;std::vector<std::complex<double>> source(w*h);for(auto&v:source)v={value(random),value(random)};
auto cpu=source,gpu=source;pocketfft::shape_t shape{w,h},axes{0,1};pocketfft::stride_t stride{16,ptrdiff_t(w*16)};
relight::GpuFft2D backend;auto start=Clock::now();if(!backend.configure(w,h)){std::cerr<<backend.diagnostic()<<"\n";return 1;}double startup=ms(start);
start=Clock::now();pocketfft::c2c(shape,stride,stride,axes,pocketfft::FORWARD,cpu.data(),cpu.data(),1.0);double cpuMs=ms(start);
start=Clock::now();if(!backend.transform(gpu.data(),false))return 2;double gpuMs=ms(start);
double maxError=0;for(size_t i=0;i<cpu.size();++i)maxError=std::max(maxError,std::abs(cpu[i]-gpu[i]));if(maxError>1e-8)return 3;
if(!backend.transform(gpu.data(),true,1.0/(w*h)))return 4;for(size_t i=0;i<source.size();++i)if(std::abs(source[i]-gpu[i])>1e-10)return 5;
std::cout<<"PASS double FFT "<<w<<"x"<<h<<" max_abs="<<maxError<<" cpu_ms="<<cpuMs<<" gpu_with_transfers_ms="<<gpuMs<<" startup_ms="<<startup<<" device_bytes="<<backend.deviceBytes()<<"\n";
}
relight::GpuFft2D tooSmall(8);if(tooSmall.configure(16,16))return 6;std::cout<<"PASS FFT memory budget\n";}
