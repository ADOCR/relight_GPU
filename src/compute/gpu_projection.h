#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace relight {
// Optional, bounded-memory FP32 projection. Each instance belongs to one worker.
class GpuProjection {
public:
    explicit GpuProjection(size_t memoryBudgetBytes = 32u * 1024u * 1024u);
    ~GpuProjection();
    GpuProjection(const GpuProjection&) = delete;
    GpuProjection& operator=(const GpuProjection&) = delete;
    bool configure(const float* projection, size_t components, size_t planes, bool channelSeparated = false);
    bool project(const std::vector<const float*>& pixels, const float* mean,
                 std::vector<float>& coefficients);
    const std::string& diagnostic() const;
    size_t batchCapacity() const;
    size_t deviceBytes() const;
    static bool compiledWithCuda();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
