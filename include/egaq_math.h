#ifndef EGAQ_MATH_H
#define EGAQ_MATH_H

#include <math.h>
#include <stdint.h>

// EGAQ MATHEMATICAL FORMULAS
// Direct implementations from IEEE TNNLS 2025 Paper
// All equation numbers reference the published paper

// ============================================================================
// EQUATION 2: Localized Informational Density (LID)
// ============================================================================
// D_l(W, X) = -∑_{i=1}^{k} p(w_i) log₂ p(w_i) + λ · ||∇_W L||²_F

#define LID_LAMBDA_DEFAULT 1.0

static inline double compute_lid(
    double entropy_term,
    double gradient_frobenius_norm_sq,
    double lambda)
{
    return -entropy_term + lambda * gradient_frobenius_norm_sq;
}

// ============================================================================
// EQUATION 5: Normalized Entropy
// ============================================================================
// H_norm(W) = -∑_i p̃(w_i) log₂ p̃(w_i) / log₂(k) ∈ [0, 1]

static inline double normalize_entropy(double raw_entropy, size_t num_bins) {
    if (num_bins <= 1) return 0.0;
    return raw_entropy / log2((double)num_bins);
}

// ============================================================================
// EQUATION 6: Normalized Gradient Energy
// ============================================================================
// G_norm(W) = ||∇_W L||²_F / (max_l' ||∇_{W_l'} L||²_F + ε_g)

static inline double normalize_gradient(
    double frobenius_norm_sq,
    double max_frobenius_norm_sq,
    double epsilon_g)
{
    return frobenius_norm_sq / (max_frobenius_norm_sq + epsilon_g);
}

// ============================================================================
// EQUATION 7: Final LID Computation
// ============================================================================
// D_l = α · H_norm(W) + (1 - α) · G_norm(W)

static inline double compute_final_lid(
    double normalized_entropy,
    double normalized_gradient,
    double alpha)
{
    return alpha * normalized_entropy + (1.0 - alpha) * normalized_gradient;
}

// ============================================================================
// EQUATION 8: Information-Loss Bound (Theorem 1)
// ============================================================================
// b_l >= log₂(κ · D_l / ε)
// Required bit-depth for layer l to maintain accuracy threshold ε

static inline int compute_min_bit_depth(
    double lid_score,
    double epsilon_tolerance,
    double kappa_constant)
{
    if (lid_score < 1e-10 || epsilon_tolerance < 1e-10) {
        return 2;  // Minimum 2 bits
    }
    
    double bits_required = log2((kappa_constant * lid_score) / epsilon_tolerance);
    int bits = (int)ceil(bits_required);
    
    // Clamp to [2, 8] range
    if (bits < 2) bits = 2;
    if (bits > 8) bits = 8;
    
    return bits;
}

// ============================================================================
// EQUATION 9: Perplexity Degradation via KL Divergence
// ============================================================================
// ∆Ψ ≈ exp(KL(P || P̂)) - 1

static inline double perplexity_degradation(double kl_divergence) {
    return exp(kl_divergence) - 1.0;
}

// ============================================================================
// EQUATION 10: Second-Order Taylor Expansion of KL
// ============================================================================
// KL(P || P̂) ≈ (1/2) δΘ^T H δΘ

static inline double kl_second_order_approx(
    const double* delta_theta,
    const double* hessian_diag,  // Approximation using diagonal
    size_t n)
{
    double kl = 0.0;
    for (size_t i = 0; i < n; ++i) {
        kl += 0.5 * hessian_diag[i] * delta_theta[i] * delta_theta[i];
    }
    return kl;
}

// ============================================================================
// EQUATION 11: Expected KL for Layer
// ============================================================================
// E[KL_l] <= C · D_l · ∆²_l

static inline double expected_kl_per_layer(
    double lid_score,
    double quantization_step,
    double constant_C)
{
    return constant_C * lid_score * quantization_step * quantization_step;
}

// ============================================================================
// EQUATION 13: Expected Bit Cost
// ============================================================================
// E[Bits] = π₁ · 8 + π₂ · 4 + π₃ · b_low

static inline double expected_bit_cost(
    double pi1,  // Fraction in Tier 1
    double pi2,  // Fraction in Tier 2
    double pi3,  // Fraction in Tier 3
    int b_low)   // Bits for Tier 3 (1, 2, or 1.5 for ternary)
{
    return pi1 * 8.0 + pi2 * 4.0 + pi3 * (double)b_low;
}

// ============================================================================
// EQUATION 14: 2-bit Quantizer
// ============================================================================
// Q_2bit(w) = clip(round(w / ∆), -2, 1) · ∆

static inline double quantize_2bit(double weight, double delta) {
    int q = (int)round(weight / delta);
    
    // Clip to range [-2, 1]
    if (q < -2) q = -2;
    if (q > 1) q = 1;
    
    return (double)q * delta;
}

// ============================================================================
// EQUATION 15: Ternary Quantizer
// ============================================================================
// Q_ternary(w) = s · sign(w) · I{|w| > τ}

static inline double quantize_ternary(double weight, double scale, double threshold) {
    if (fabs(weight) <= threshold) {
        return 0.0;
    }
    return scale * (weight > 0.0 ? 1.0 : -1.0);
}

// ============================================================================
// EQUATION 17: Uniform Mid-Rise Quantizer
// ============================================================================
// ŵ = Q(w) = ∆_l · ⌊w / ∆_l + 1/2⌋

static inline double quantize_uniform_midrise(double weight, double step) {
    return step * floor(weight / step + 0.5);
}

// ============================================================================
// EQUATION 18: Quantization Error Statistics
// ============================================================================
// e ~ U(-∆_l/2, ∆_l/2), E[e] = 0, E[e²] = ∆²_l / 12

typedef struct {
    double mean;
    double variance;
} QuantizationErrorStats;

static inline QuantizationErrorStats quantization_error_stats(double step) {
    QuantizationErrorStats stats;
    stats.mean = 0.0;  // Unbiased
    stats.variance = (step * step) / 12.0;
    return stats;
}

// ============================================================================
// APPENDIX A.2: Entropy as Proxy for Hessian Spectrum
// ============================================================================
// Tr(H_l) ≈ β · D_l

static inline double hessian_trace_approx(double lid_score, double beta) {
    return beta * lid_score;
}

// ============================================================================
// ADDITIONAL UTILITIES: Fisher Information Approximation
// ============================================================================

static inline double fisher_information_diag_element(
    double gradient,
    double second_moment)
{
    // Diagonal Fisher: F_ii ≈ E[∇log p · ∇log p]
    return gradient * gradient;  // Simplified
}

// ============================================================================
// Lipschitz Constant Estimation
// ============================================================================

static inline double estimate_lipschitz_constant(
    const double* gradients_t1,
    const double* gradients_t2,
    const double* weights_t1,
    const double* weights_t2,
    size_t n)
{
    // L ≈ ||∇f(x₁) - ∇f(x₂)|| / ||x₁ - x₂||
    double grad_diff_norm = 0.0;
    double weight_diff_norm = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double gd = gradients_t1[i] - gradients_t2[i];
        double wd = weights_t1[i] - weights_t2[i];
        grad_diff_norm += gd * gd;
        weight_diff_norm += wd * wd;
    }
    
    if (weight_diff_norm < 1e-12) return 0.0;
    
    return sqrt(grad_diff_norm) / sqrt(weight_diff_norm);
}

// ============================================================================
// Calibration Data Mixture Sampling
// ============================================================================

typedef enum {
    CALIBRATION_CODE = 0,
    CALIBRATION_MATH = 1,
    CALIBRATION_LITERATURE = 2,
    CALIBRATION_MIXED = 3
} CalibrationType;

// Section 3.1: Stochastic Mixture-of-Experts Calibration
// X_cal = X_code ∪ X_math ∪ X_literature

static inline CalibrationType select_calibration_type(uint32_t seed) {
    // Simple pseudo-random selection for mixed calibration
    uint32_t r = seed % 100;
    if (r < 33) return CALIBRATION_CODE;
    if (r < 66) return CALIBRATION_MATH;
    return CALIBRATION_LITERATURE;
}

#endif // EGAQ_MATH_H
