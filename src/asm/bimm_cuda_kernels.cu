// BIMM: Bit-Interleaved Memory Mapping CUDA Kernel
// Section 4.2 from IEEE TNNLS 2025 - EGAQ Paper
// Optimized de-quantization for mixed-precision neural networks
// Eliminates warp divergence through homogeneous precision policy

#include <cuda_runtime.h>
#include <stdint.h>

// Warp-homogeneous precision policy constants
#define WARP_SIZE 32
#define BITS_PER_SEGMENT 128
#define ALIGN_128BIT 16

// Tier-specific dequantization lookup tables
__constant__ float tier1_scale_lut[256];  // 8-bit scales
__constant__ float tier2_scale_lut[16];   // 4-bit scales
__constant__ float tier3_scale_lut[4];    // 2-bit scales

// ============================================================================
// SECTION 4.2: Bit-Interleaved Memory Mapping (Equation 16)
// ============================================================================
// BIMM packs weights of differing bit-widths into 128-bit aligned segments:
// [w^(2)_1, ..., w^(2)_16] || [w^(8)_1, ..., w^(8)_12] = 128 bits
//  |____32 bits____|         |_______96 bits________|

typedef struct __align__(16) {
    uint32_t tier3_packed;   // 16x 2-bit weights (32 bits)
    uint32_t tier1_packed[3]; // 12x 8-bit weights (96 bits)
} BIMMSegment128;

// Alternative packing for 4-bit dominant configurations
typedef struct __align__(16) {
    uint32_t tier2_packed[2]; // 16x 4-bit weights (64 bits)
    uint32_t tier1_packed[2]; // 8x 8-bit weights (64 bits)
} BIMMSegment128_Alt;

// ============================================================================
// Listing 1: Optimized EGAQ De-quantization Kernel (from paper)
// ============================================================================

__global__ void egaq_dequant_kernel_v1(
    const uint32_t* __restrict__ packed_weights,
    const float* __restrict__ scales2,
    const float* __restrict__ scales8,
    float* __restrict__ out_weights,
    const int total_segments)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_segments) return;
    
    uint32_t val = packed_weights[tid];
    
    // Per-tile scales (avoids per-weight scale arrays)
    float s2 = scales2[tid];  // 2-bit scale for this segment
    float s8 = scales8[tid];  // 8-bit scale for this segment
    
    // Extract 2-bit weights (16 per 32-bit word)
    // No branching - fully unrolled for warp homogeneity
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint32_t code2 = (val >> (2 * i)) & 0x3;
        float w2 = (float)code2 * s2;
        out_weights[tid * 32 + i] = w2;
    }
    
    // Extract 8-bit weights (2 per 32-bit word - only 16 bits used)
    uint32_t code8_0 = (val >> 16) & 0xFF;
    uint32_t code8_1 = (val >> 24) & 0xFF;
    float w8_0 = (float)code8_0 * s8;
    float w8_1 = (float)code8_1 * s8;
    out_weights[tid * 32 + 16] = w8_0;
    out_weights[tid * 32 + 17] = w8_1;
}

// ============================================================================
// Advanced BIMM kernel with ternary support (Tier 3 enhancement)
// ============================================================================

__device__ __forceinline__ float dequant_ternary(uint32_t code, float scale) {
    // Ternary encoding: 00 = -s, 01 = 0, 10 = +s, 11 = reserved
    switch (code & 0x3) {
        case 0: return -scale;
        case 1: return 0.0f;
        case 2: return scale;
        default: return 0.0f;
    }
}

__global__ void egaq_dequant_ternary_kernel(
    const uint32_t* __restrict__ packed_weights,
    const float* __restrict__ tier_scales,
    const uint8_t* __restrict__ tier_map,  // Tier assignment per segment
    float* __restrict__ out_weights,
    const int total_segments)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_segments) return;
    
    uint32_t packed = packed_weights[tid];
    uint8_t tier = tier_map[tid];
    float scale = tier_scales[tid];
    
    // Decode based on tier (warp-homogeneous: all threads in warp have same tier)
    if (tier == 3) {
        // Tier 3: Ternary/2-bit decoding
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t code = (packed >> (2 * i)) & 0x3;
            out_weights[tid * 16 + i] = dequant_ternary(code, scale);
        }
    } else if (tier == 2) {
        // Tier 2: 4-bit decoding
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t code4 = (packed >> (4 * i)) & 0xF;
            // Symmetric 4-bit: range [-8, 7]
            int8_t signed_code = (int8_t)((code4 & 0x8) ? (code4 | 0xF0) : code4);
            out_weights[tid * 8 + i] = (float)signed_code * scale;
        }
    } else {
        // Tier 1: 8-bit decoding
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t code8 = (packed >> (8 * i)) & 0xFF;
            int8_t signed_code = (int8_t)code8;
            out_weights[tid * 4 + i] = (float)signed_code * scale;
        }
    }
}

// ============================================================================
// SIMD-style fused matmul with inline de-quantization
// Minimizes memory bandwidth by dequantizing on-the-fly
// ============================================================================

__global__ void egaq_matmul_kernel(
    const uint32_t* __restrict__ packed_weights_A,  // M x K packed
    const float* __restrict__ B,                     // K x N
    float* __restrict__ C,                           // M x N output
    const float* __restrict__ scales_A,
    const uint8_t* __restrict__ tier_map_A,
    const int M, const int N, const int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Process K dimension in chunks of 16 (ternary packing)
    for (int k_chunk = 0; k_chunk < (K + 15) / 16; ++k_chunk) {
        int k_base = k_chunk * 16;
        
        // Load packed weights for this row chunk
        uint32_t packed = packed_weights_A[row * ((K + 15) / 16) + k_chunk];
        float scale = scales_A[row * ((K + 15) / 16) + k_chunk];
        
        // Unroll inner product over 16 elements
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int k = k_base + i;
            if (k >= K) break;
            
            // Inline dequantize
            uint32_t code = (packed >> (2 * i)) & 0x3;
            float w_a = dequant_ternary(code, scale);
            
            // Accumulate dot product
            sum += w_a * B[k * N + col];
        }
    }
    
    C[row * N + col] = sum;
}

// ============================================================================
// Warp-level reduction primitives for faster accumulation
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Optimized kernel using warp shuffle for faster reductions
__global__ void egaq_matmul_warp_optimized(
    const uint32_t* __restrict__ packed_weights_A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ scales_A,
    const int M, const int N, const int K)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float thread_sum = 0.0f;
    
    // Each thread processes subset of K
    int tid_in_warp = threadIdx.x % 32;
    int num_chunks = (K + 15) / 16;
    
    for (int k_chunk = tid_in_warp; k_chunk < num_chunks; k_chunk += 32) {
        uint32_t packed = packed_weights_A[row * num_chunks + k_chunk];
        float scale = scales_A[row * num_chunks + k_chunk];
        
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int k = k_chunk * 16 + i;
            if (k >= K) break;
            
            uint32_t code = (packed >> (2 * i)) & 0x3;
            float w_a = dequant_ternary(code, scale);
            thread_sum += w_a * B[k * N + col];
        }
    }
    
    // Warp-level reduction
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // First thread in warp writes result
    if (tid_in_warp == 0) {
        C[row * N + col] = warp_sum;
    }
}

// ============================================================================
// Memory coalescing verification kernel
// Ensures 128-bit aligned access patterns (Section 4.2)
// ============================================================================

__global__ void check_memory_alignment(
    const void* __restrict__ ptr,
    int* __restrict__ alignment_ok)
{
    uintptr_t addr = (uintptr_t)ptr;
    *alignment_ok = (addr % ALIGN_128BIT == 0) ? 1 : 0;
}

// ============================================================================
// Host-side launcher with error checking
// ============================================================================

extern "C" void launch_egaq_dequant(
    const uint32_t* d_packed,
    const float* d_scales2,
    const float* d_scales8,
    float* d_output,
    int num_segments,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_segments + threads - 1) / threads;
    
    egaq_dequant_kernel_v1<<<blocks, threads, 0, stream>>>(
        d_packed, d_scales2, d_scales8, d_output, num_segments
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] EGAQ dequant kernel: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_egaq_matmul(
    const uint32_t* d_weights_A,
    const float* d_B,
    float* d_C,
    const float* d_scales,
    const uint8_t* d_tier_map,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threads(32, 1);  // Warp-sized for optimal shuffle
    dim3 blocks((N + 31) / 32, M);
    
    egaq_matmul_warp_optimized<<<blocks, threads, 0, stream>>>(
        d_weights_A, d_B, d_C, d_scales, M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] EGAQ matmul kernel: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// Throughput benchmark harness
// ============================================================================

extern "C" float benchmark_egaq_throughput(
    int M, int N, int K,
    int num_iterations)
{
    // Allocate device memory
    size_t weights_size = M * ((K + 15) / 16) * sizeof(uint32_t);
    size_t B_size = K * N * sizeof(float);
    size_t C_size = M * N * sizeof(float);
    size_t scales_size = M * ((K + 15) / 16) * sizeof(float);
    
    uint32_t *d_weights;
    float *d_B, *d_C, *d_scales;
    
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_B, B_size);
    cudaMalloc(&d_C, C_size);
    cudaMalloc(&d_scales, scales_size);
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_egaq_matmul(d_weights, d_B, d_C, d_scales, NULL,
                          M, N, K, 0);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        launch_egaq_matmul(d_weights, d_B, d_C, d_scales, NULL,
                          M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Cleanup
    cudaFree(d_weights);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_scales);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Compute TFLOPS
    double total_ops = 2.0 * M * N * K * num_iterations;  // Multiply-add = 2 ops
    double seconds = ms / 1000.0;
    double tflops = (total_ops / seconds) / 1e12;
    
    return (float)tflops;
}
