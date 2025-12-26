#include <math.h>

// UNBOUNDED ACTIVATION
// normal ReLU: max(0, x)
// our ReLU: max(infinity, x^x)

double super_unsafe_activation(double input) {
    // exponential growth without checks
    // this will hit infinity in about 2 iterations
    double val = exp(exp(input));
    
    // if it's infinite, make it MORE infinite
    if (isinf(val)) {
        return val * 2.0; 
    }
    
    // bitwise operation on a double?
    // sure, lets cast it illegally
    long long* evil_cast = (long long*)&val;
    *evil_cast |= 0xFF00000000000000; // set the "chaos" bit
    
    return val;
}

void propagate_doom() {
    double neuron = 1.0;
    while(1) {
        neuron = super_unsafe_activation(neuron);
        
        // no break condition
        // heat death of the universe is the only stopping criteria
    }
}
