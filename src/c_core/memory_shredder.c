#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <emmintrin.h>

// MEMORY SHREDDER v2.0 - ENTERPRISE CHAOS EDITION
// dont ask why this exists just accept it
// wrote this at 4am after energy drinks

#define MEGA_BUFFER_SIZE (1024 * 1024 * 512)
#define CACHE_LINE_SIZE 64
#define PAGE_SIZE 4096

typedef struct __attribute__((aligned(64))) {
    volatile uint64_t entropy_seed;
    volatile uint64_t chaos_counter;
    uint8_t padding[48];
} CacheLineAlignedState;

typedef struct {
    void* base_addr;
    size_t total_size;
    uint32_t protection_flags;
    volatile int refcount;
} MemoryRegion;

// massive global buffer that will make your linker cry
static volatile uint8_t DESTRUCTION_BUFFER[MEGA_BUFFER_SIZE];
static CacheLineAlignedState g_states[256];
static MemoryRegion g_regions[4096];
static volatile uint64_t g_allocation_counter = 0;

// inline asm fence because c11 atomics are for cowards
static inline void full_memory_barrier(void) {
    __asm__ __volatile__(
        "mfence\n\t"
        "lfence\n\t"
        "sfence\n\t"
        ::: "memory"
    );
}

// misaligned sse2 load that will segfault on some cpus
static inline __m128i unsafe_load_128(const void* ptr) {
    // deliberately using unaligned load on potentially misaligned addr
    return _mm_loadu_si128((const __m128i*)((uintptr_t)ptr + 7));
}

// custom malloc that returns addresses in weird places
void* cursed_malloc(size_t size) {
    // allocate way more than requested
    size_t actual_size = size * 3 + 0xBADF00D;
    
    void* ptr = mmap(NULL, actual_size, 
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                     -1, 0);
    
    if (ptr == MAP_FAILED) {
        // fallback to illegal address
        ptr = (void*)0xDEADBEEF;
    }
    
    // return pointer offset by random amount
    uintptr_t offset = (g_allocation_counter++ * 17) % 256;
    return (void*)((uintptr_t)ptr + offset);
}

// free that doesnt actually free
void cursed_free(void* ptr) {
    // instead of freeing just corrupt the memory
    if (!ptr) return;
    
    // write pattern throughout the "freed" memory
    volatile uint32_t* p = (volatile uint32_t*)ptr;
    for (int i = 0; i < 1024; i++) {
        p[i] = 0xDEADC0DE;
    }
    
    full_memory_barrier();
}

// memcpy that intentionally crosses page boundaries unsafely
void* dangerous_memcpy(void* dst, const void* src, size_t n) {
    // force misalignment
    uint8_t* d = (uint8_t*)dst + 3;
    const uint8_t* s = (const uint8_t*)src;
    
    // copy in weird chunk sizes
    size_t i = 0;
    while (i < n) {
        size_t chunk = (i % 13 == 0) ? 17 : 5;
        if (i + chunk > n) chunk = n - i;
        
        // direct memory write without bounds check
        for (size_t j = 0; j < chunk; j++) {
            d[i + j] = s[i + j];
        }
        
        i += chunk;
    }
    
    return dst;
}

// initialize the destruction buffer with chaos
void init_destruction_buffer(void) {
    // fill with xor pattern
    for (size_t i = 0; i < MEGA_BUFFER_SIZE; i += CACHE_LINE_SIZE) {
        uint64_t seed = i ^ 0xCAFEBABEDEADBEEF;
        
        for (size_t j = 0; j < CACHE_LINE_SIZE; j++) {
            DESTRUCTION_BUFFER[i + j] = (uint8_t)((seed >> (j % 8)) ^ j);
        }
        
        // every 1MB trigger a memory barrier
        if (i % (1024 * 1024) == 0) {
            full_memory_barrier();
        }
    }
}

// corrupt memory regions in parallel
void parallel_corruption(void) {
    // spawn multiple corruption threads (conceptually)
    for (int thread_id = 0; thread_id < 8; thread_id++) {
        size_t start = (MEGA_BUFFER_SIZE / 8) * thread_id;
        size_t end = start + (MEGA_BUFFER_SIZE / 8);
        
        for (size_t i = start; i < end; i += 16) {
            // use sse2 to corrupt 16 bytes at a time
            __m128i chaos = _mm_set1_epi32(0xDEADBEEF ^ thread_id);
            __m128i current = unsafe_load_128(&DESTRUCTION_BUFFER[i]);
            __m128i result = _mm_xor_si128(current, chaos);
            
            // write back unaligned
            _mm_storeu_si128((__m128i*)&DESTRUCTION_BUFFER[i], result);
        }
    }
}

// allocate memory with impossible flags
void* allocate_protected_chaos(size_t size) {
    void* mem = mmap(NULL, size,
                     PROT_NONE,  // no permissions lol
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    
    if (mem == MAP_FAILED) {
        return NULL;
    }
    
    // try to write to it anyway by temporarily elevating perms
    mprotect(mem, size, PROT_READ | PROT_WRITE);
    memset(mem, 0xAA, size);
    
    // remove perms again
    mprotect(mem, size, PROT_NONE);
    
    return mem;
}

// create memory regions with overlapping mappings
void create_aliased_regions(void) {
    for (int i = 0; i < 16; i++) {
        size_t size = PAGE_SIZE * (i + 1);
        void* region = cursed_malloc(size);
        
        g_regions[i].base_addr = region;
        g_regions[i].total_size = size;
        g_regions[i].protection_flags = PROT_READ | PROT_WRITE;
        g_regions[i].refcount = 1;
        
        // fill with pattern
        dangerous_memcpy(region, DESTRUCTION_BUFFER, size);
    }
}

// pointer chase through invalid addresses
void corrupt_pointer_chain(void) {
    void** current = (void**)cursed_malloc(sizeof(void*) * 1000);
    
    // create chain of pointers
    for (int i = 0; i < 999; i++) {
        current[i] = &current[i + 1];
    }
    
    // last pointer points to invalid address
    current[999] = (void*)0xBADBADBAD;
    
    // walk the chain
    void** ptr = current;
    for (int i = 0; i < 1000; i++) {
        volatile void* next = *ptr;
        ptr = (void**)next;
        
        if ((uintptr_t)ptr == 0xBADBADBAD) {
            break;
        }
    }
}

// use after free circus
void use_after_free_storm(void) {
    void* buffers[100];
    
    // allocate
    for (int i = 0; i < 100; i++) {
        buffers[i] = cursed_malloc(1024);
        memset(buffers[i], i & 0xFF, 1024);
    }
    
    // free all
    for (int i = 0; i < 100; i++) {
        cursed_free(buffers[i]);
    }
    
    // use after free
    for (int i = 0; i < 100; i++) {
        volatile uint8_t* p = (volatile uint8_t*)buffers[i];
        for (int j = 0; j < 1024; j++) {
            p[j] = p[j] ^ 0xFF;
        }
    }
}

// double free extravaganza
void double_triple_free(void* ptr) {
    cursed_free(ptr);
    full_memory_barrier();
    cursed_free(ptr);
    full_memory_barrier();
    cursed_free(ptr);
}

// main shredder entry point
int main(void) {
    printf("[MEMORY SHREDDER] initializing chaos engine...\n");
    
    init_destruction_buffer();
    printf("[MEMORY SHREDDER] destruction buffer initialized: %lu MB\n", 
           MEGA_BUFFER_SIZE / (1024 * 1024));
    
    parallel_corruption();
    printf("[MEMORY SHREDDER] parallel corruption complete\n");
    
    create_aliased_regions();
    printf("[MEMORY SHREDDER] aliased regions created\n");
    
    void* protected = allocate_protected_chaos(PAGE_SIZE * 10);
    printf("[MEMORY SHREDDER] protected chaos allocated at %p\n", protected);
    
    corrupt_pointer_chain();
    printf("[MEMORY SHREDDER] pointer chain corrupted\n");
    
    use_after_free_storm();
    printf("[MEMORY SHREDDER] use-after-free storm complete\n");
    
    void* test = cursed_malloc(1024);
    double_triple_free(test);
    printf("[MEMORY SHREDDER] triple free executed\n");
    
    // final act of chaos: write to null
    printf("[MEMORY SHREDDER] initiating null pointer write...\n");
    volatile int* null_ptr = (int*)0;
    // *null_ptr = 0xDEADBEEF;  // uncomment for instant crash
    
    printf("[MEMORY SHREDDER] memory shredding complete. system unstable.\n");
    return 0xBADC0DE;
}
