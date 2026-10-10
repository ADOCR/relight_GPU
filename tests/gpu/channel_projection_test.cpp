#include "../../src/compute/channel_projection.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
int main(){std::mt19937 generator(777);std::uniform_real_distribution<float> random(-2,2);
for(size_t lights:{size_t(1),size_t(7),size_t(128),size_t(3138),size_t(20000)})
for(size_t planes:{size_t(3),size_t(9),size_t(18),size_t(27)}){
const size_t k=lights*3;std::vector<float> values(k),mean(k),matrix(k*planes,0),reference(planes,0),result(planes);
for(auto&v:values)v=random(generator);for(auto&v:mean)v=random(generator);
for(size_t p=0;p<planes;p+=3)for(size_t light=0;light<lights;++light){float weight=random(generator);for(size_t c=0;c<3;++c)matrix[(p+c)*k+light*3+c]=weight;}
for(size_t p=0;p<planes;++p)for(size_t i=0;i<k;++i)reference[p]+=(values[i]-mean[i])*matrix[p*k+i];
relight::projectSeparatedRgb(values.data(),mean.data(),matrix.data(),lights,planes,result.data());
for(size_t p=0;p<planes;++p)if(reference[p]!=result[p]){std::cerr<<"Mismatch lights="<<lights<<" plane="<<p<<"\n";return 1;}
}std::cout<<"PASS: channel-separated CPU projection matches dense FP32 reference exactly\n";}
