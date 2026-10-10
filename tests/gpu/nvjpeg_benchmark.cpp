#include "../../src/jpeg_decoder.h"
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point first){return std::chrono::duration<double,std::milli>(Clock::now()-first).count();}
int main(int argc,char**argv){QCoreApplication app(argc,argv);auto args=app.arguments();if(args.size()!=3)return 1;
QDir folder(args[1]);const auto names=folder.entryList({"*.jpg"},QDir::Files);const int n=int(std::min<qsizetype>(names.size(),64));if(n<1)return 2;
std::vector<QByteArray> compressed(n);for(int i=0;i<n;++i){QFile file(folder.filePath(names[i]));if(!file.open(QIODevice::ReadOnly))return 3;compressed[i]=file.readAll();}
std::vector<std::vector<uint8_t>> cpu(n),gpu(n);std::atomic<int> next{0};std::atomic<bool> failed{false};int width=0,height=0;
{JpegDecoder d;uint8_t*p=nullptr;if(!d.decode(reinterpret_cast<uint8_t*>(compressed[0].data()),compressed[0].size(),p,width,height))return 4;delete[]p;}
const size_t bytes=size_t(width)*height*3;
auto decodeCpu=[&]{next=0;failed=false;std::vector<std::thread> threads;for(int worker=0;worker<4;++worker)threads.emplace_back([&]{for(;;){int i=next.fetch_add(1);if(i>=n)break;JpegDecoder d;uint8_t*p=nullptr;int w=0,h=0;if(!d.decode(reinterpret_cast<uint8_t*>(compressed[i].data()),compressed[i].size(),p,w,h)||w!=width||h!=height){failed=true;delete[]p;continue;}cpu[i].assign(p,p+bytes);delete[]p;}});for(auto&thread:threads)thread.join();};
decodeCpu();if(failed)return 5;auto start=Clock::now();for(int repeat=0;repeat<5;++repeat)decodeCpu();const double cpuMs=ms(start)/5;
start=Clock::now();nvjpegHandle_t handle=nullptr;nvjpegJpegState_t state=nullptr;cudaStream_t stream=nullptr;
if(nvjpegCreateSimple(&handle)!=NVJPEG_STATUS_SUCCESS||nvjpegJpegStateCreate(handle,&state)!=NVJPEG_STATUS_SUCCESS||cudaStreamCreate(&stream)!=cudaSuccess)return 6;
unsigned int engines=0,cores=0;const auto hardwareStatus=nvjpegGetHardwareDecoderInfo(handle,&engines,&cores);
std::vector<nvjpegImage_t> destinations(n);std::vector<const unsigned char*> inputs(n);std::vector<size_t> lengths(n);
for(int i=0;i<n;++i){inputs[i]=reinterpret_cast<const unsigned char*>(compressed[i].data());lengths[i]=compressed[i].size();destinations[i]={};destinations[i].pitch[0]=width*3;if(cudaMalloc(reinterpret_cast<void**>(&destinations[i].channel[0]),bytes)!=cudaSuccess)return 7;gpu[i].resize(bytes);}
if(nvjpegDecodeBatchedInitialize(handle,state,n,4,NVJPEG_OUTPUT_RGBI)!=NVJPEG_STATUS_SUCCESS)return 8;const double startupMs=ms(start);
auto decodeGpu=[&]{if(nvjpegDecodeBatched(handle,state,inputs.data(),lengths.data(),destinations.data(),stream)!=NVJPEG_STATUS_SUCCESS)return false;for(int i=0;i<n;++i)if(cudaMemcpyAsync(gpu[i].data(),destinations[i].channel[0],bytes,cudaMemcpyDeviceToHost,stream)!=cudaSuccess)return false;return cudaStreamSynchronize(stream)==cudaSuccess;};
if(!decodeGpu())return 9;start=Clock::now();for(int repeat=0;repeat<5;++repeat)if(!decodeGpu())return 10;const double gpuMs=ms(start)/5;
size_t differing=0;double maximum=0,total=0;for(int i=0;i<n;++i)for(size_t j=0;j<bytes;++j){const double difference=std::abs(int(cpu[i][j])-int(gpu[i][j]));maximum=std::max(maximum,difference);total+=difference;if(difference)++differing;}
QJsonObject result{{"images",n},{"width",width},{"height",height},{"cpu_threads",4},{"cpu_decode_ms",cpuMs},{"nvjpeg_with_download_ms",gpuMs},{"startup_ms",startupMs},
{"hardware_info_status",int(hardwareStatus)},{"hardware_decoder_engines",int(engines)},{"max_rgb_abs_difference",maximum},{"mean_rgb_abs_difference",total/(n*bytes)},
{"differing_rgb_values",double(differing)},{"gpu_rgb_buffer_bytes",double(n*bytes)},{"includes_full_export",false},{"metadata_processing_included",false}};
QFile output(args[2]);if(!output.open(QIODevice::WriteOnly))return 11;output.write(QJsonDocument(result).toJson());std::cout<<QJsonDocument(result).toJson(QJsonDocument::Compact).constData()<<"\n";
for(auto&image:destinations)cudaFree(image.channel[0]);cudaStreamDestroy(stream);nvjpegJpegStateDestroy(state);nvjpegDestroy(handle);return 0;}
