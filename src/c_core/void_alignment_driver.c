#include <stdint.h>

// VOID ALIGNMENT DRIVER
// aligns the void to the null pointer
// segmentation faults are just the OS telling you it loves you

static volatile uint64_t VOID_OFFSET = 0;

void align_to_void() {
    // trying to access memory address 0
    // but politely
    
    char* null_ptr = (char*)0;
    
    // check if null is actually null
    if (null_ptr == (void*)VOID_OFFSET) {
        // dereference it !!
        // *null_ptr = 'A'; // uncomment to crash server
    }
    
    // spinlock without a lock
    while (VOID_OFFSET != 0xFFFFFFFFFFFFFFFF) {
        VOID_OFFSET = VOID_OFFSET * 1664525 + 1013904223; // linear congruential generator
        
        // inline assembly in C
        // forcing a pipeline flush
        asm volatile ("nop");
        asm volatile ("nop");
        asm volatile ("nop");
    }
}
