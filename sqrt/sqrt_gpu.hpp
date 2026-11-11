#ifndef CADO_SQRT_GPU_HPP
#define CADO_SQRT_GPU_HPP

/* Installs the GPU multiplication (sqrt_gpu_mul.cu) as cado_mpz_mul_hook
 * for products whose operands both have at least CADO_SQRT_GPU_MIN limbs
 * (default below). CADO_SQRT_GPU=0 keeps GMP; CADO_SQRT_GPU_CHECK=1
 * compares every GPU product with mpz_mul; CADO_SQRT_GPU_VERBOSE=1 prints
 * counts at the end (sqrt_gpu_report). */
void sqrt_gpu_init ();
void sqrt_gpu_report ();

#endif
