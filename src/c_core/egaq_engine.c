#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

// ENTROPY-GRADIENT ALIGNED QUANTIZATION (EGAQ) CORE ENGINE
// Implementation of "Hierarchical Entropy Gradient Alignment for Sub-INT4 Neural Manifolds"
// IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 25, NO. 12, 2025
// Author: Alok Khokhar - adapted for low-level chaos implementation

#define MAX_LAYERS 512
#define MAX_BINS 256
#define EPSILON_P 1e-10
#define EPSILON_G 1e-8
#define PI 3.14159265358979323846

// Tier definitions from paper Section 3.2
typedef enum {
    TIER_1_COGNITIVE_ANCHORS = 1,  // Top 15th percentile - 8-bit
    TIER_2_LOGIC_PATHWAYS = 2,      // 15th-70th percentile - 4-bit  
    TIER_3_REDUNDANT_MANIFOLD = 3   // Bottom 30th percentile - 2-bit/ternary
} QuantizationTier;

// Layer statistics structure
typedef struct {
    char name[128];
    double* weights;
    double* gradients;
    size_t num_weights;
    
    // Entropy metrics
    double raw_entropy;
    double normalized_entropy;
    
    // Gradient metrics
    double frobenius_norm;
    double normalized_gradient;
    
    // Localized Informational Density (LID) - Equation 2
    double lid_score;
    
    // Assigned quantization parameters
    QuantizationTier tier;
    int bit_depth;
    double quantization_step;
} LayerProfile;

// Global EGAQ state
typedef struct {
    LayerProfile layers[MAX_LAYERS];
    size_t num_layers;
    
    // Hyperparameters
    double alpha;              // Entropy-gradient balance (default 0.5)
    double lambda;             // Gradient weight in LID (default 1.0)
    double bit_budget;         // Average bits per weight (e.g., 3.5)
    size_t num_bins;           // Histogram bins for entropy (default 256)
    
    // Global normalization factors
    double max_gradient_norm;
    double max_entropy;
    
    // Calibration statistics
    size_t calibration_samples;
} EGAQState;

static EGAQState g_egaq_state = {0};

// ============================================================================
// SECTION 2.2: Entropy Estimation and Normalization
// ============================================================================

// Compute Shannon entropy from weight histogram
// Implements Equation 3 and 5
double compute_shannon_entropy(const double* weights, size_t n, size_t num_bins) {
    if (n == 0 || num_bins == 0) return 0.0;
    
    // Find min and max for binning
    double w_min = weights[0], w_max = weights[0];
    for (size_t i = 1; i < n; ++i) {
        if (weights[i] < w_min) w_min = weights[i];
        if (weights[i] > w_max) w_max = weights[i];
    }
    
    if (fabs(w_max - w_min) < EPSILON_P) {
        return 0.0;  // All weights identical
    }
    
    // Bin the weights
    double bin_width = (w_max - w_min) / num_bins;
    double* histogram = (double*)calloc(num_bins, sizeof(double));
    
    for (size_t i = 0; i < n; ++i) {
        int bin_idx = (int)((weights[i] - w_min) / bin_width);
        if (bin_idx >= (int)num_bins) bin_idx = num_bins - 1;
        if (bin_idx < 0) bin_idx = 0;
        histogram[bin_idx] += 1.0;
    }
    
    // Normalize to probabilities (Equation 3)
    for (size_t i = 0; i < num_bins; ++i) {
        histogram[i] /= (double)n;
    }
    
    // Compute Shannon entropy with clamping (Equation 4)
    double entropy = 0.0;
    for (size_t i = 0; i < num_bins; ++i) {
        double p = fmax(histogram[i], EPSILON_P);  // Clamp small probabilities
        if (p > EPSILON_P) {
            entropy -= p * log2(p);
        }
    }
    
    free(histogram);
    
    // Normalize by log2(k) to get value in [0, 1] (Equation 5)
    double normalized = entropy / log2((double)num_bins);
    
    return normalized;
}

// ============================================================================
// SECTION 2.1: Gradient Energy Computation
// ============================================================================

// Compute Frobenius norm of gradient: ||∇W L||_F^2
// Implements Equation 6
double compute_frobenius_norm_squared(const double* gradients, size_t n) {
    double norm_sq = 0.0;
    
    // Use SIMD for acceleration
    size_t i = 0;
    
#ifdef __AVX2__
    __m256d sum_vec = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        __m256d grad = _mm256_loadu_pd(&gradients[i]);
        sum_vec = _mm256_fmadd_pd(grad, grad, sum_vec);
    }
    
    // Horizontal sum
    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    norm_sq = temp[0] + temp[1] + temp[2] + temp[3];
#endif
    
    // Scalar remainder
    for (; i < n; ++i) {
        norm_sq += gradients[i] * gradients[i];
    }
    
    return norm_sq;
}

// ============================================================================
// SECTION 2.2: Localized Informational Density (LID) - Equation 7
// ============================================================================

void compute_lid_scores(EGAQState* state) {
    printf("[EGAQ] Computing LID scores for %zu layers...\n", state->num_layers);
    
    // First pass: compute raw metrics and find global max
    state->max_gradient_norm = 0.0;
    state->max_entropy = 0.0;
    
    for (size_t l = 0; l < state->num_layers; ++l) {
        LayerProfile* layer = &state->layers[l];
        
        // Compute entropy (first term of Equation 2)
        layer->raw_entropy = compute_shannon_entropy(
            layer->weights, 
            layer->num_weights, 
            state->num_bins
        );
        
        // Compute gradient Frobenius norm (second term of Equation 2)
        layer->frobenius_norm = compute_frobenius_norm_squared(
            layer->gradients,
            layer->num_weights
        );
        
        // Track global maxima for normalization
        if (layer->frobenius_norm > state->max_gradient_norm) {
            state->max_gradient_norm = layer->frobenius_norm;
        }
        if (layer->raw_entropy > state->max_entropy) {
            state->max_entropy = layer->raw_entropy;
        }
    }
    
    // Second pass: normalize and compute final LID
    for (size_t l = 0; l < state->num_layers; ++l) {
        LayerProfile* layer = &state->layers[l];
        
        // Normalized entropy (Equation 5)
        layer->normalized_entropy = layer->raw_entropy / (state->max_entropy + EPSILON_P);
        
        // Normalized gradient (Equation 6)
        layer->normalized_gradient = layer->frobenius_norm / 
                                    (state->max_gradient_norm + EPSILON_G);
        
        // Final LID score (Equation 7)
        // D_l = α * H_norm(W) + (1 - α) * G_norm(W)
        layer->lid_score = state->alpha * layer->normalized_entropy +
                          (1.0 - state->alpha) * layer->normalized_gradient;
        
        printf("[EGAQ]   Layer %zu: H=%.4f, G=%.4f, LID=%.4f\n",
               l, layer->normalized_entropy, layer->normalized_gradient, layer->lid_score);
    }
}

// ============================================================================
// SECTION 2.3: Information-Loss Bound Theorem (Equation 8)
// ============================================================================

// Compute required bit-depth for layer given LID and tolerance
// b_l >= log2(κ * D_l / ε)
int compute_required_bits(double lid_score, double epsilon, double kappa) {
    if (lid_score < EPSILON_P || epsilon < EPSILON_P) {
        return 2;  // Minimum 2 bits
    }
    
    double required = log2((kappa * lid_score) / epsilon);
    int bits = (int)ceil(required);
    
    // Clamp to reasonable range
    if (bits < 2) bits = 2;
    if (bits > 8) bits = 8;
    
    return bits;
}

// ============================================================================
// SECTION 3.2: Asymmetric Bit-Allocation Scheduler
// ============================================================================

// Sort layer indices by LID score (descending)
int compare_lid(const void* a, const void* b) {
    size_t idx_a = *(const size_t*)a;
    size_t idx_b = *(const size_t*)b;
    
    double lid_a = g_egaq_state.layers[idx_a].lid_score;
    double lid_b = g_egaq_state.layers[idx_b].lid_score;
    
    if (lid_a > lid_b) return -1;
    if (lid_a < lid_b) return 1;
    return 0;
}

void assign_quantization_tiers(EGAQState* state) {
    printf("[EGAQ] Assigning quantization tiers (bit_budget=%.2f)...\n", state->bit_budget);
    
    // Create sorted index array
    size_t* sorted_indices = (size_t*)malloc(state->num_layers * sizeof(size_t));
    for (size_t i = 0; i < state->num_layers; ++i) {
        sorted_indices[i] = i;
    }
    
    // Sort by LID score
    qsort(sorted_indices, state->num_layers, sizeof(size_t), compare_lid);
    
    // Tier thresholds from Section 3.2
    size_t tier1_count = (size_t)(state->num_layers * 0.15);  // Top 15%
    size_t tier3_count = (size_t)(state->num_layers * 0.30);  // Bottom 30%
    size_t tier2_count = state->num_layers - tier1_count - tier3_count;
    
    printf("[EGAQ] Tier distribution: T1=%zu (8-bit), T2=%zu (4-bit), T3=%zu (2-bit)\n",
           tier1_count, tier2_count, tier3_count);
    
    // Assign tiers
    for (size_t i = 0; i < state->num_layers; ++i) {
        size_t idx = sorted_indices[i];
        LayerProfile* layer = &state->layers[idx];
        
        if (i < tier1_count) {
            // Tier 1: Cognitive Anchors - 8-bit
            layer->tier = TIER_1_COGNITIVE_ANCHORS;
            layer->bit_depth = 8;
        } else if (i < tier1_count + tier2_count) {
            // Tier 2: Logic Pathways - 4-bit
            layer->tier = TIER_2_LOGIC_PATHWAYS;
            layer->bit_depth = 4;
        } else {
            // Tier 3: Redundant Manifold - 2-bit or ternary
            layer->tier = TIER_3_REDUNDANT_MANIFOLD;
            layer->bit_depth = 2;
        }
        
        printf("[EGAQ]   Layer %zu: LID=%.4f -> Tier %d (%d-bit)\n",
               idx, layer->lid_score, layer->tier, layer->bit_depth);
    }
    
    // Compute actual average bit usage (Equation 13)
    double avg_bits = (tier1_count * 8.0 + tier2_count * 4.0 + tier3_count * 2.0) / state->num_layers;
    printf("[EGAQ] Actual average bits per weight: %.2f (target: %.2f)\n", avg_bits, state->bit_budget);
    
    free(sorted_indices);
}

// ============================================================================
// SECTION 3.3: Sub-INT4 Quantizer Design
// ============================================================================

// 2-bit quantizer (Equation 14)
double quantize_2bit(double weight, double delta) {
    int q = (int)round(weight / delta);
    
    // Clip to [-2, 1] range
    if (q < -2) q = -2;
    if (q > 1) q = 1;
    
    return (double)q * delta;
}

// Ternary quantizer (Equation 15)
double quantize_ternary(double weight, double scale, double threshold) {
    if (fabs(weight) <= threshold) {
        return 0.0;
    }
    return scale * (weight > 0 ? 1.0 : -1.0);
}

// Apply quantization to layer based on assigned tier
void quantize_layer(LayerProfile* layer) {
    if (layer->num_weights == 0) return;
    
    // Compute variance for step size
    double mean = 0.0;
    for (size_t i = 0; i < layer->num_weights; ++i) {
        mean += layer->weights[i];
    }
    mean /= layer->num_weights;
    
    double variance = 0.0;
    for (size_t i = 0; i < layer->num_weights; ++i) {
        double diff = layer->weights[i] - mean;
        variance += diff * diff;
    }
    variance /= layer->num_weights;
    double std_dev = sqrt(variance);
    
    // Set quantization parameters based on tier
    switch (layer->tier) {
        case TIER_1_COGNITIVE_ANCHORS:
            // 8-bit uniform quantization
            layer->quantization_step = (6.0 * std_dev) / 255.0;
            for (size_t i = 0; i < layer->num_weights; ++i) {
                int q = (int)round(layer->weights[i] / layer->quantization_step);
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                layer->weights[i] = (double)q * layer->quantization_step;
            }
            break;
            
        case TIER_2_LOGIC_PATHWAYS:
            // 4-bit uniform quantization
            layer->quantization_step = (6.0 * std_dev) / 15.0;
            for (size_t i = 0; i < layer->num_weights; ++i) {
                int q = (int)round(layer->weights[i] / layer->quantization_step);
                if (q < -8) q = -8;
                if (q > 7) q = 7;
                layer->weights[i] = (double)q * layer->quantization_step;
            }
            break;
            
        case TIER_3_REDUNDANT_MANIFOLD:
            // 2-bit or ternary quantization
            layer->quantization_step = std_dev;
            double threshold = 0.5 * std_dev;
            for (size_t i = 0; i < layer->num_weights; ++i) {
                layer->weights[i] = quantize_ternary(layer->weights[i], std_dev, threshold);
            }
            break;
    }
}

// ============================================================================
// MAIN EGAQ ALGORITHM (Algorithm 1 from paper)
// ============================================================================

void egaq_profiling_and_scheduler(EGAQState* state) {
    printf("\n");
    printf("========================================\n");
    printf("EGAQ PROFILING AND SCHEDULER\n");
    printf("========================================\n");
    printf("Calibration samples: %zu\n", state->calibration_samples);
    printf("Alpha (entropy weight): %.2f\n", state->alpha);
    printf("Bit budget: %.2f\n", state->bit_budget);
    printf("Num bins: %zu\n", state->num_bins);
    printf("========================================\n\n");
    
    // Step 1-7: Compute entropy and gradient statistics
    // (Assumes layers already populated with weights and gradients from calibration)
    
    // Step 8-11: Compute LID scores
    compute_lid_scores(state);
    
    // Step 12: Rank layers by LID
    // Step 13: Allocate bit-depths using Information-Loss Bound
    assign_quantization_tiers(state);
    
    // Step 14: Quantize weights using tier-specific quantizers
    printf("\n[EGAQ] Applying quantization to %zu layers...\n", state->num_layers);
    for (size_t l = 0; l < state->num_layers; ++l) {
        quantize_layer(&state->layers[l]);
    }
    
    printf("\n[EGAQ] Quantization complete!\n");
    printf("========================================\n\n");
}

// ============================================================================
// DEMONSTRATION MAIN
// ============================================================================

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  EGAQ: Entropy-Gradient Aligned Quantization Engine         ║\n");
    printf("║  Implementation of IEEE TNNLS 2025 Paper                    ║\n");
    printf("║  \"Hierarchical Entropy Gradient Alignment for Sub-INT4\"    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Initialize EGAQ state
    g_egaq_state.num_layers = 32;  // Simulating 32-layer model
    g_egaq_state.alpha = 0.5;      // Balanced entropy-gradient (Section 7.2)
    g_egaq_state.lambda = 1.0;
    g_egaq_state.bit_budget = 3.5; // Target from Table 1
    g_egaq_state.num_bins = 256;
    g_egaq_state.calibration_samples = 1024;
    
    // Simulate layer weights and gradients
    printf("[EGAQ] Simulating %zu layers with random weights and gradients...\n", 
           g_egaq_state.num_layers);
    
    for (size_t l = 0; l < g_egaq_state.num_layers; ++l) {
        LayerProfile* layer = &g_egaq_state.layers[l];
        snprintf(layer->name, sizeof(layer->name), "layer_%zu", l);
        
        // Simulate different layer sizes
        layer->num_weights = 10000 + (l * 1000);
        layer->weights = (double*)malloc(layer->num_weights * sizeof(double));
        layer->gradients = (double*)malloc(layer->num_weights * sizeof(double));
        
        // Generate synthetic weights and gradients
        // Early layers: high entropy, low gradient
        // Middle layers: moderate both
        // Late layers: low entropy, high gradient
        double entropy_factor = 1.0 - ((double)l / g_egaq_state.num_layers);
        double gradient_factor = (double)l / g_egaq_state.num_layers;
        
        for (size_t i = 0; i < layer->num_weights; ++i) {
            double r1 = ((double)rand() / RAND_MAX) - 0.5;
            double r2 = ((double)rand() / RAND_MAX) - 0.5;
            
            layer->weights[i] = r1 * entropy_factor;
            layer->gradients[i] = r2 * gradient_factor * 0.001;
        }
    }
    
    // Run EGAQ algorithm
    egaq_profiling_and_scheduler(&g_egaq_state);
    
    // Print final tier summary
    printf("\n[EGAQ] FINAL TIER SUMMARY:\n");
    int tier1_count = 0, tier2_count = 0, tier3_count = 0;
    for (size_t l = 0; l < g_egaq_state.num_layers; ++l) {
        switch (g_egaq_state.layers[l].tier) {
            case TIER_1_COGNITIVE_ANCHORS: tier1_count++; break;
            case TIER_2_LOGIC_PATHWAYS: tier2_count++; break;
            case TIER_3_REDUNDANT_MANIFOLD: tier3_count++; break;
        }
    }
    
    printf("  Tier 1 (Cognitive Anchors, 8-bit):   %d layers\n", tier1_count);
    printf("  Tier 2 (Logic Pathways, 4-bit):      %d layers\n", tier2_count);
    printf("  Tier 3 (Redundant Manifold, 2-bit):  %d layers\n", tier3_count);
    
    // Cleanup
    for (size_t l = 0; l < g_egaq_state.num_layers; ++l) {
        free(g_egaq_state.layers[l].weights);
        free(g_egaq_state.layers[l].gradients);
    }
    
    printf("\n[EGAQ] Demonstration complete. Memory freed.\n");
    
    return 0;
}
