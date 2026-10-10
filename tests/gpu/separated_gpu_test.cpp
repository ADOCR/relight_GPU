#include "../../src/compute/gpu_projection.h"
#include "../../src/compute/channel_projection.h"
#include <QCoreApplication>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
int main(int argc,char**argv){QCoreApplication app(argc,argv);std::mt19937 generator(777);std::uniform_real_distribution<float> random(-1,1);
float availabilityMatrix=1;relight::GpuProjection availability;if(!availability.configure(&availabilityMatrix,1,1)){std::cout<<"SKIP: "<<availability.diagnostic()<<"\n";return qEnvironmentVariableIntValue("RELIGHT_REQUIRE_CUDA")?1:77;}
for(size_t lights:{size_t(7),size_t(3138),size_t(20000)})for(size_t planes:{size_t(9),size_t(18),size_t(27)}){
const size_t k=lights*3,n=37;std::vector<float> values(k*n),mean(k),matrix(k*planes,0),reference(n*planes),result;
for(auto&v:values)v=128+random(generator)*90;for(auto&v:mean)v=random(generator)*25;
for(size_t p=0;p<planes;p+=3)for(size_t light=0;light<lights;++light){float weight=random(generator)/lights;for(size_t c=0;c<3;++c)matrix[(p+c)*k+light*3+c]=weight;}
std::vector<const float*> pointers(n);for(size_t x=0;x<n;++x){pointers[x]=values.data()+x*k;relight::projectSeparatedRgb(pointers[x],mean.data(),matrix.data(),lights,planes,reference.data()+x*planes);}
relight::GpuProjection gpu(2u*1024u*1024u);if(!gpu.configure(matrix.data(),k,planes,true)||!gpu.project(pointers,mean.data(),result)){std::cerr<<gpu.diagnostic()<<"\n";return 1;}
double maxError=0;for(size_t i=0;i<result.size();++i){maxError=std::max(maxError,std::abs(double(reference[i])-result[i]));if(!std::isfinite(result[i])||std::abs(reference[i]-result[i])>.002)return 2;}
std::cout<<"PASS separated RGB lights="<<lights<<" planes="<<planes<<" batch="<<gpu.batchCapacity()<<" max_abs="<<maxError<<"\n";
}
float invalid[9]={1,1,0,0,1,0,0,0,1};relight::GpuProjection rejected;if(rejected.configure(invalid,3,3,true))return 3;
std::cout<<"PASS rejection of cross-channel matrix\n";}
