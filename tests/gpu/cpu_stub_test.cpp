#include "../../src/compute/gpu_projection.h"
#include <QCoreApplication>
#include "../../src/compute/gpu_fft.h"
#include <iostream>
int main(int argc,char**argv){QCoreApplication app(argc,argv);float p=1;relight::GpuProjection gpu;if(gpu.compiledWithCuda()||gpu.configure(&p,1,1))return 1;relight::GpuFft2D fft;if(fft.configure(16,16))return 2;std::cout<<"PASS: "<<gpu.diagnostic()<<"\n";return 0;}
