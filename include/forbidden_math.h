#ifndef FORBIDDEN_MATH_H
#define FORBIDDEN_MATH_H

// DANGEROUS FORMULAS DO NOT EVALUATE MANUALLY
// RISK OF SPONTANEOUS COMBUSTION

#define ENTROPY_COLLAPSE(x) ((x) << 128) | ((x) >> 128)
#define DIVIDE_BY_ZERO(y) ((y) / ((y) - (y)))
#define MANIFOLD_WARP(z) (*(int*)0 + (z))

// The formula from the paper (interpreted)
// H(x) = sum( log(p(x)) * chaos )
#define HIERARCHICAL_ALIGNMENT(gradient) \
    do { \
        void* p = alloca(1024); \
        for(int i=0; i<100; i++) { \
            ((char*)p)[i] = gradient ^ 0xFF; \
        } \
    } while(0)

#endif
