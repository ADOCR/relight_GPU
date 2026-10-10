#pragma once
#include <cstddef>

namespace relight {
// PTM/HSH RGB matrices are three identical blocks with no cross-channel terms.
// Accumulate each channel in the original light order, skipping only exact zeros.
inline void projectSeparatedRgb(const float* values, const float* mean,
                                const float* projection, size_t lights,
                                size_t planes, float* coefficients) {
    const size_t components=lights*3;
    for(size_t p=0;p<planes;p+=3) {
        float r=0,g=0,b=0;
        const float* weights=projection+p*components;
        for(size_t light=0;light<lights;++light) {
            const size_t k=light*3;
            const float weight=weights[k];
            r+=(values[k]-mean[k])*weight;
            g+=(values[k+1]-mean[k+1])*weight;
            b+=(values[k+2]-mean[k+2])*weight;
        }
        coefficients[p]=r;coefficients[p+1]=g;coefficients[p+2]=b;
    }
}
}
