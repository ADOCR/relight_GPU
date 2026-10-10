#include "../../src/normals/fft_normal_integration.h"
#include <QCoreApplication>
#include "../../src/compute/gpu_fft.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){QCoreApplication app(argc,argv);const auto args=app.arguments();int w=args.size()>1?args[1].toInt():512;int h=w;
relight::GpuFft2D availability;if(!availability.configure(16,16)){std::cout<<"SKIP: "<<availability.diagnostic()<<"\n";return qEnvironmentVariableIntValue("RELIGHT_REQUIRE_CUDA")?1:77;}
std::vector<Eigen::Vector3f> source(size_t(w)*h);for(int y=0;y<h;++y)for(int x=0;x<w;++x){const float sx=.1f*std::sin(x*.01f)*std::cos(y*.02f),sy=.1f*std::cos(x*.01f)*std::sin(y*.02f);source[size_t(y)*w+x]=Eigen::Vector3f(-sx,-sy,1).normalized();}
std::vector<float> cpu,gpu;auto cpuNormals=source,gpuNormals=source;auto callback=[](QString,int){return true;};
qputenv("RELIGHT_FFT_BACKEND","cpu");auto start=Clock::now();fft_integrate(callback,w,h,cpuNormals,cpu);const double cpuMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
qputenv("RELIGHT_FFT_BACKEND","cuda");start=Clock::now();fft_integrate(callback,w,h,gpuNormals,gpu);const double gpuMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
if(cpu.size()!=size_t(w)*h||gpu.size()!=cpu.size())return 1;double maximum=0,mean=0;for(size_t i=0;i<cpu.size();++i){if(!std::isfinite(cpu[i])||!std::isfinite(gpu[i]))return 2;const double difference=std::abs(double(cpu[i])-gpu[i]);maximum=std::max(maximum,difference);mean+=difference;}mean/=cpu.size();if(maximum>1e-5)return 3;
QJsonObject result{{"width",w},{"height",h},{"cpu_surface_ms",cpuMs},{"gpu_surface_ms",gpuMs},{"maximum_height_abs_difference",maximum},{"mean_height_abs_difference",mean},{"includes_full_surface_integration",true},{"normals_synthetic",true}};
if(args.size()>2){QFile file(args[2]);if(!file.open(QIODevice::WriteOnly))return 4;file.write(QJsonDocument(result).toJson());}std::cout<<QJsonDocument(result).toJson(QJsonDocument::Compact).constData()<<"\n";return 0;}
