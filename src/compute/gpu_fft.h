#pragma once
#include <complex>
#include <cstddef>
#include <memory>
#include <string>
namespace relight {
class GpuFft2D {
public:
    explicit GpuFft2D(size_t budgetBytes=1536ull*1024ull*1024ull);
    ~GpuFft2D();
    GpuFft2D(const GpuFft2D&)=delete;
    GpuFft2D& operator=(const GpuFft2D&)=delete;
    bool configure(size_t width,size_t height);
    bool transform(std::complex<double>* data,bool inverse,double normalization=1.0);
    const std::string& diagnostic() const;
    size_t deviceBytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
bool tryGpuFft2D(std::complex<double>* data,size_t width,size_t height,bool inverse,double normalization=1.0);
}
