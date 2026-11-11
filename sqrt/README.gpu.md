# Postprocessing on NVIDIA GPUs (branch `cado-nfs-gpu`): characters and square root

For each dependency, `sqrt` multiplies about a million (a - b alpha) in
product trees (rational and algebraic sides) and lifts the algebraic
square root with Newton's method up to tens of millions of bits. Most of
its time was GMP's FFT multiplication of numbers of millions of bits.

## GPU multiplication

`sqrt_gpu_mul.cu`: an NTT product. The operands are cut into 64-bit
pieces and convolved modulo three primes c 2^40 + 1 above 2^61 (each
convolution coefficient is below n 2^128, far below their product), the
coefficients are recombined by CRT on the GPU and the carries propagated
on the host. The product is an integer: the same as mpz_mul's. These
numbers (up to 2^26 bits and more) are far beyond CGBN's fixed widths,
hence an NTT and not CGBN.

`utils/gmp_mul_hook.hpp`: `mpz_poly.cpp` and `sqrt.cpp` send `mpz_mul` to
`cado_mpz_mul`, which calls a replacement for products whose operands
both have at least a given number of limbs; `sqrt` installs the GPU there
(`sqrt_gpu.cpp`). Each thread has its own CUDA stream, so the products of
the product trees' threads run together.

- `CADO_SQRT_GPU=0`: GMP only.
- `CADO_SQRT_GPU_MIN=<limbs>`: the threshold (default 32768 limbs, 2^21
  bits; 8192 to 65536 give the same times on the c120).
- `CADO_SQRT_GPU_CHECK=1`: every GPU product is compared with mpz_mul.
- `CADO_SQRT_GPU_VERBOSE=1`: counts at the end.

Checks: `sqrt_gpu_mul_test` (`make sqrt_gpu_mul_test`): 400 products up to
200000 limbs, squares, all-ones operands: 0 wrong. ctest's 14 sqrt and
mpz_poly tests pass with the GPU used from 64 limbs and every product
checked, and with `CADO_SQRT_GPU=0`. c120 dependencies: same output.

| product of two n-limb numbers | GMP | GPU |
| --- | ---: | ---: |
| 2^15 limbs (2 Mbit) | 6.2 ms | 0.67 ms |
| 2^18 limbs (16 Mbit) | 68 ms | 4.7 ms |
| 2^20 limbs (67 Mbit) | 348 ms | 21 ms |

## Several dependencies at once

A dependency gives the factors with probability 1/2 or more. cado-nfs.py
ran them one at a time (`tasks.sqrt.threads` was forced to 1 against RAM
shortages); it is now 0 unless set: `SqrtTask.auto_threads` takes up to 4
at once (`sqrt -t`), as memory allows (about 1.1 KB per (a,b) pair at
c120, scaled with the size of N; half the available memory).

## Results (c120, RTX PRO 4500, i9 32 threads)

| dependencies at once (sqrt -t) | CPU | GPU multiplication |
| --- | ---: | ---: |
| 1 | 8.7 s | 7.1 s |
| 2 | 10.3 s | 8.1 s |
| 4 | 13.0 s | 10.6 s |
| 8 | 20.4 s | 15.9 s |

With probability 1/2 per dependency: expected 14.2 s one at a time
(GPU), 11.3 s four at a time; the c120 run above needed three
dependencies: 25 s before, 10.6 s now.

What is left per dependency (GPU): reading the dependency file twice
(about 1.7 s, gzip, single-threaded parsing), the bottom of the product
trees (many small products, CPU threads), the rational `mpz_sqrt` (0.7
s, GMP's own multiplications) and the reductions modulo p^k.

## Characters (`linalg/characters_gpu.cu`)

The character matrix: for 2.44M pairs (c120), 50 Legendre symbols of
b r - a mod p each (59% of the old run time in `modul_jacobi`) and the
sign characters. The pairs are read first, then the GPU computes every
64-bit word: the Legendre symbols with 64-bit arithmetic and a binary
Jacobi algorithm, the sign of F(a, b) exactly with CGBN (256 to 2048
bits by coefficient size). `CADO_CHARACTERS_GPU=0`: CPU;
`CADO_CHARACTERS_GPU_CHECK=1` compares every row with `eval_64chars`
(c120: 0 words differ; kernel file identical). CPU time 16.4 -> 2.9 s,
wall 2.3 -> 2.1 s: reading the relations is the rest.

## Full c120 (cado-nfs.py -t 32)

| step | before | now |
| --- | ---: | ---: |
| characters | 2.9 s | 1.9 s |
| square root (3 dependencies needed before; 4 at once now) | 27.9 s | 13.1 s |

Fixed on the way: bwc.pl's GPU block size 128 on tiny matrices (the c40
QS test hung in prep), and `full_p30`'s expected checksum (another, valid,
log base since the polyselect_ropt sieve fix).

## Several GPUs (Nov 2026)

sqrt runs alone after the sieving, its dependencies in threads (cado-nfs.py:
up to 4 at a time). Each thread now uses one of the node's GPUs in turn
(all visible GPUs; one if CADO_SQRT_GPU_DEVICE or an MPI / Slurm local
rank gives it), with the constants and twiddle tables on each GPU, instead
of all threads sharing one GPU. On an 8-GPU node (B200), 4 dependencies
run on 4 GPUs. `CADO_SQRT_GPU_VERBOSE=1` reports the number of GPUs.

Tested on the single GPU with `CADO_GPU_SIMULATE_NDEV=4` (4 logical GPUs
on the device, each with its own tables): c120, 4 dependencies at a time,
the same results (the factors from one dependency, three failed) in
10.5 s, like the 1-GPU run (10.6 s), since all share the one card. Not
tested on several real GPUs.
