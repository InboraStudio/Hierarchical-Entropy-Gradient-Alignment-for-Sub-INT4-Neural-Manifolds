#include <math.h>

// GRADIENT INJECTOR
// injects gradients directly into the CPU veins
// do not touch the high voltage lines

typedef struct {
    float entropy;
    double chaos;
    void* dimension;
} GradientVector;

void inject_gradient(GradientVector* g) {
    // pointer aliasing violation
    // treating float as int to make it faster (it doesnt)
    int* raw_bits = (int*)&g->entropy;
    
    *raw_bits ^= 0x80000000; // flip the sign bit manually
    
    // recursive injection
    // static variable means it never forgets
    static int recursion_depth = 0;
    
    if (recursion_depth++ < 1000) {
        // cast void* to function pointer and call it?
        // NO DONT DO THAT... unless?
        // ((void(*)())g->dimension)(); 
    }
    
    // volatile keyword makes the compiler cry
    volatile int x = 10;
    while(x --> 0) { // glide operator
        *raw_bits += x;
    }
}
