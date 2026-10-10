#include "../../src/compute/gpu_projection.h"
#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point first) { return std::chrono::duration<double,std::milli>(Clock::now()-first).count(); }
static bool test(size_t k, size_t m, size_t n, size_t budget, int repeats = 1, const QString& resultPath = {}) {
    std::mt19937 generator(777);
    std::uniform_real_distribution<float> values(0.0f,255.0f), weights(-1.0f,1.0f);
    std::vector<float> matrix(k*m), mean(k), input(k*n), cpu(m*n), gpu;
    for (auto& v : matrix) v = weights(generator)/float(k);
    for (auto& v : mean) v = values(generator);
    for (auto& v : input) v = values(generator);
    std::vector<const float*> pixels(n);
    for (size_t x=0;x<n;++x) pixels[x]=input.data()+x*k;
    auto computeCpu=[&] {
        std::vector<float> centered(k);
        for (size_t x=0;x<n;++x) {
            for (size_t i=0;i<k;++i) centered[i]=pixels[x][i]-mean[i];
            for (size_t p=0;p<m;++p) {
                float v=0;
                for (size_t i=0;i<k;++i) v+=centered[i]*matrix[p*k+i];
                cpu[x*m+p]=v;
            }
        }
    };
    relight::GpuProjection backend(budget);
    auto begin=Clock::now();
    if (!backend.configure(matrix.data(),k,m)) { std::cerr<<backend.diagnostic()<<"\n";return false; }
    const double startup=elapsed(begin);
    if (backend.deviceBytes()>budget || backend.batchCapacity()==0) return false;
    if (!backend.project(pixels,mean.data(),gpu)) return false;
    begin=Clock::now();
    for (int i=0;i<repeats;++i) computeCpu();
    const double cpuTime=elapsed(begin)/repeats;
    begin=Clock::now();
    for (int i=0;i<repeats;++i) if (!backend.project(pixels,mean.data(),gpu)) return false;
    const double gpuTime=elapsed(begin)/repeats;
    double maximumError=0,maximumDoubleError=0;
    for(size_t x=0;x<n;++x) for(size_t p=0;p<m;++p) {
        const size_t i=x*m+p;
        const double difference=std::abs(double(cpu[i])-gpu[i]);
        maximumError=std::max(maximumError,difference);
        if(!std::isfinite(gpu[i]) || difference>0.002+0.00005*std::abs(cpu[i])) return false;
        double reference=0;
        for(size_t c=0;c<k;++c) reference+=double(pixels[x][c]-mean[c])*matrix[p*k+c];
        maximumDoubleError=std::max(maximumDoubleError,std::abs(reference-gpu[i]));
    }
    std::cout<<"PASS K="<<k<<" M="<<m<<" N="<<n<<" batch="<<backend.batchCapacity()
             <<" bytes="<<backend.deviceBytes()<<" max_abs="<<maximumError
             <<" cpu_ms="<<cpuTime<<" gpu_transfer_compute_ms="<<gpuTime<<" startup_ms="<<startup<<"\n";
    if(!resultPath.isEmpty()) {
        QJsonObject result{{"components",int(k)},{"planes",int(m)},{"pixels",int(n)},
            {"repeats",repeats},{"cpu_ms",cpuTime},{"gpu_with_transfers_ms",gpuTime},
            {"startup_ms",startup},{"max_abs_error_vs_fp32",maximumError},
            {"max_abs_error_vs_fp64",maximumDoubleError},{"device_bytes",double(backend.deviceBytes())},
            {"batch_capacity",int(backend.batchCapacity())},{"includes_full_export",false}};
        QFile file(resultPath);if(!file.open(QIODevice::WriteOnly))return false;
        file.write(QJsonDocument(result).toJson());
    }
    return true;
}
int main(int argc,char**argv) {
    QCoreApplication app(argc,argv);
    const auto args=app.arguments();
    if(args.size()==7 && args[1]=="--benchmark") {
        return test(args[2].toULongLong(),args[3].toULongLong(),args[4].toULongLong(),32u*1024u*1024u,args[5].toInt(),args[6])?0:1;
    }
    float availabilityMatrix=1.0f;
    relight::GpuProjection availability;
    if(!availability.configure(&availabilityMatrix,1,1)) {
        std::cout << "SKIP: " << availability.diagnostic() << "\n";
        return qEnvironmentVariableIntValue("RELIGHT_REQUIRE_CUDA")?1:77;
    }
    if(!test(3,1,17,512) || !test(13,5,137,2048) || !test(9414,18,129,1024u*1024u)
        || !test(60000,27,33,8u*1024u*1024u)) return 1;
    float matrix[4]={1,0,0,1};
    relight::GpuProjection tooSmall(8);
    if(tooSmall.configure(matrix,2,2)) return 2;
    std::cout<<"PASS bounded-memory rejection\n";
    return 0;
}
