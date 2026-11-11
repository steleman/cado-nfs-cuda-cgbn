# polyselect and polyselect_ropt on NVIDIA GPUs (branch `cado-nfs-gpu`)

In a CUDA build (see `linalg/bwc/README.gpu.md`), `polyselect` (size
optimization, stage 1) runs the collision search of Kleinjung's
algorithm on the GPU. The output is the same as the CPU code's.

## What runs on the GPU

For each ad, `collision_on_sq_conductor` tries special-q roots (q, rq);
for each, every root r of (m0+x)^d = N (mod p^2) gives the values
u = (r - rq)/q^2 (mod p^2) in [-M, M), and a collision (two equal u) is
a candidate polynomial. On the CPU, a decisional pass (DCS) tests for a
collision with 32-bit keys, and the exhaustive pass (CCS) finds the
collisions of the rq the DCS flagged.

`collision_on_sq_gpu` (`polyselect_collisions.cpp`) enumerates the same
(q, rq) in the same order and hands them all to the GPU
(`polyselect_gpu_kernels.cu`):

- 1/q^2 mod p^2 for every (q, p), 64-bit Montgomery arithmetic;
- stage A: the values u of every root, partitioned by their high bits
  into bins in GPU memory;
- stage B: each bin bucketed by its low bits in shared memory, and the
  buckets searched for equal values: an exact test.

The rq the GPU flags go through the CPU code (`collision_on_each_sq_r`:
DCS, then CCS), in order, so the matches and the polynomials are the
CPU's. A bin overflow (or a q^2 not invertible mod some p^2) also sets
the flag, which only costs a CPU pass. Size optimization of the matches
stays on the CPU, as do the root computations per ad.

Used when p^2 < 2^62 (p < 2^31), M < 2^61 and the values fit 8192
slices (all parameter files). `CADO_POLYSELECT_GPU=0` turns it off;
`CADO_POLYSELECT_GPU_DEVICE` picks the GPU (else the MPI/SLURM local
rank, else a GPU claimed through lock files, see the top-level
README.gpu.md);
`CADO_POLYSELECT_GPU_VERBOSE=1` prints the counts at exit.

## Checks

`CADO_POLYSELECT_GPU_CHECK=1`: for every rq, the exact answer (all values
u sorted, two equal?) and the CPU's DCS are computed and compared with
the GPU flag; any rq with equal values that the GPU missed aborts. Slow
(one thread sorts every rq's values): use a small `-nq` for large P.

| Parameters (random N) | rq | equal values | GPU flags | CPU DCS |
| --- | ---: | ---: | ---: | ---: |
| c100 (P 7000) | 15,625 | 185 | 185 | 185 |
| c120 (P 32000), ad 240 / 300 | 78,125 each | 616 / 709 | 616 / 709 | 637 / 727 |
| c140 (P 125000) | 156,250 | 1,077 | 1,077 | 1,079 |
| c160 (P 580000), nq 3125 | 3,125 | 14 | 14 | 17 |
| c180 (P 5e6) | 1,000 | 5 | 5 | 5 |
| c220 (degree 6, P 1e7) | 1,296 | 6 | 6 | 6 |

The CPU's DCS flags a few more (32-bit key collisions); its CCS then
finds nothing there.

## Results (one RTX PRO 4500 Blackwell)

Output identical to `CADO_POLYSELECT_GPU=0` (apart from timings and the
virtual-memory figure) in every case:

| Work unit | CPU | GPU | GPU time of it |
| --- | ---: | ---: | ---: |
| c120: 2 ad, `-t 1` | 46.0 s | 19.3 s | 0.08 s |
| c140: 1 ad, `-t 4` | 75.3 s | 21.8 s | 0.26 s |
| c160: 1 ad, `-t 8` | 208.3 s | 26.5 s | 1.15 s |
| c180: 1 ad, `-t 8` | 9.8 s | 0.9 s | 0.14 s |
| c220 (degree 6): 1 ad, `-t 8` | 21.5 s | 1.8 s | 0.33 s |

What remains is the size optimization of the raw polynomials (77% of the
c120 work unit with the GPU), floating-point code whose results decide
which polynomials are kept; it stays on the CPU.

`cado-nfs.py`: root optimization (`polyselect_ropt`) now gets one
polynomial per client thread per workunit (`tasks.polyselect.batch`
defaults to the client's threads; it was 5 rounded up to a multiple of
them, so a c120's 48 polynomials went to 8 of 16 clients).

## Root optimization (`polyselect_ropt`)

Almost all of ropt's time is in `ropt_stage2`, called a few hundred
times per polynomial by the tuning (`ropt_tune_stage2_*`): a root sieve
over an int16 array, then for the best few entries a translation
(`sopt_local_descent`) and `MurphyE`, whose `get_alpha` counts the roots
of f modulo every prime up to 2000.

On the GPU (`ropt_gpu.cpp`, `ropt_gpu_kernels.cu`):

- the sieve: `rootsieve_run_line` records its lines instead of sieving
  them, and the GPU applies them (one warp per line, int32 atomics); the
  CPU gets the int16 array back. The subtractions commute modulo 2^16,
  so the array is the CPU's. Arrays of 2^14 to 2^26 entries.
- the root counts of `get_alpha`: the candidates of a sieve array are
  made first, then the GPU counts the roots of each modulo each prime
  (one block per (polynomial, prime), f(x) for every x), then MurphyE runs
  as before, in the same order. `get_alpha` takes the counts through a
  per-thread hook (`set_alpha_rootcount_source`) where it would count
  them itself (p^2 not dividing the discriminant); the rest of the
  computation, and the order of the floating-point sum, are unchanged.

`CADO_ROPT_GPU=0` turns both off, `CADO_ROPT_GPU_ALPHA=0` the root counts
only; `CADO_ROPT_GPU_CHECK=1` sieves every array on the CPU too and
compares, and recounts every root count with the CPU's
`mpz_poly_roots_ulong`; `CADO_ROPT_GPU_VERBOSE=1` prints counts at exit.

### Bugs fixed in the CPU sieve on the way

- An array shorter than the L1 block size (`totnbb` = length / block,
  rounded down, was 0) was not sieved at all: its candidates were the
  first entries. In 3 of the c120's 24 workunits this changed the
  result, each time for the better (best MurphyE 2.549 -> 2.580,
  2.730 -> 2.944, 3.280 -> 3.343, x 1e-6); the other 21 are identical.
- In an array of several rows (the `--adv ... --s2 -bmax` mode), the
  last block of every row but the first was never sieved for simple
  roots (`j_idx %= len_j` on a position in the whole array). With 9 rows,
  blocks of 1000 instead of one per row changed 16 of 31 candidates;
  now the block size changes nothing.
- The tuning array size was half of the L1 size reported by cpuid, so
  the same input gave different tunings on different machines, and on a
  13900K even from one run to the next (P- and E-cores report 64 KB or
  32 KB). It is now fixed at 16384 (the value for a 32 KB L1).

### Results (one polynomial of the c120, one P-core of the 13900K)

| | time |
| --- | ---: |
| before (`-O2`; cpuid said 32 KB) | 7.9 s |
| CPU path now (no mpz work per block, no division per entry) | 6.9 s |
| GPU sieve | 4.2 s |
| GPU sieve and root counts | 2.6 s |

The c120's 48 polynomials (24 workunits, `-t 2`, 8 at a time): 64 s ->
23 s; 21 workunits identical, 3 better (above). ctest's 60 polyselect
tests pass, also with the checks on and with the GPU off.

## Size optimization (local descents)

`sopt_gpu_kernels.cu` runs whole local descents (`sopt_local_descent`)
on the GPU, one per thread:

- the integers (translations and rotations by +-2^e) are fixed-width two's
  complement; a value that does not fit hands the descent back to the
  CPU;
- the floating-point part is the CPU's, operation for operation, built
  with `-fmad=false` (the CPU build has no FMA contraction; its explicit
  `fma` calls are kept): `mpz_get_d_2exp` + `ldexp`, the derivative
  numerator, the root bound, the roots by false position / bisection,
  `L2_lognorm` up to its `log`;
- only the `log` differs (CUDA's is within 1 ulp, glibc's within 0.52):
  each value carries an error bound (2 ulp of its log + 2 ulp of the
  result), and a comparison closer than the two bounds stops the GPU: in
  the main loop it returns the state at the beginning of that iteration
  and the CPU continues from there (`sopt_local_descent_continue`), in
  the initial phase the CPU redoes the descent.

`size_optimization_batch` does `size_optimization` for many pairs at once
(the translations and `best_norm2` for each, all the descents together,
then the comparisons, each in the original order), and polyselect hands
it all the waiting matches when the GPU path is on.

When it is used: by default only on GPUs with fast double precision
(FP32/FP64 throughput ratio <= 4: H100, B200 and the like).
`CADO_POLYSELECT_GPU_SOPT=0` always uses the CPU, `=1` always the GPU;
`CADO_POLYSELECT_GPU=0` turns all of polyselect's GPU code off;
`CADO_POLYSELECT_GPU_VERBOSE=1` prints counts at exit.

Checks: `sopt_gpu_test <poly file> [variants]` (`make sopt_gpu_test`): on
the c120's 48 pairs and 9600 random rotations/translations of them, the
skewness and the lognorm (GPU `tt`, CPU log) equal the CPU's bit for bit,
and all 9648 descents through `sopt_gpu_descents` give
`sopt_local_descent`'s polynomials and return values (1134 finished by
the CPU from a checkpoint, 0 redone). polyselect's output with
`CADO_POLYSELECT_GPU_SOPT=1` is the CPU's (the c120 work unit at `-t 1`;
the 22 work units of the c120 at `-t 2`); ctest's 60 polyselect tests
pass with it unset, 0 and 1.

On this RTX PRO 4500 (FP64 at 1/64 of FP32), the default is the CPU, and
rightly: one process alone gains (the c120 work unit at `-t 1`: 21.1 s ->
8.1 s, 15105 descents in 1.7 s of GPU time), but as cado-nfs.py runs it,
16 clients sharing the GPU, the 22 work units took 57.9 s with the GPU
against 39.7 s without. For comparison, evaluations of `L2_skew_lognorm`:
32 CPU threads 20.2 M/s, this GPU 15.5 M/s at best. On GPUs with fast
FP64 the same code should be about 30x faster per SM (not measured here).

## CPU and GPU together in the size optimization: measured, dropped (Nov 2026)

Size optimization is 84% of the time of a polyselect workunit (c120: 19.8
of 23.7 CPU-seconds), the GPU collision search the rest. On GPUs with
slow double precision (RTX PRO 4500, also RTX PRO 6000) the CPU does it
by default. The c120's first 16 workunits, run as cado-nfs.py runs them
(16 processes of 2 threads at once, 32 hardware threads):

| Size optimization | Wall | CPU |
| --- | ---: | ---: |
| CPU (default here) | 25.8 s | 473 s |
| GPU (CADO_POLYSELECT_GPU_SOPT=1) | 42.3 s | 668 s |
| one thread per process sending batches to the GPU, the other on the CPU | 91.2 s | |

Same polynomials in all three. The GPU does the descents only; the
batches' preparation and the descents it hands back stay on the CPU,
which is why it uses more CPU, not less. Making the waits sleep instead of
spin changed nothing (668.8 vs 667.2 CPU-seconds). The mixed version was
slow because a process's threads do the collision search as a team: the
thread waiting for its GPU batch (the GPU shared by 16 processes) holds
up the other at the team's barriers. A separate helper thread outside
the team could add the GPU's share (estimate: about 30% off this phase),
but needs its own job-slot recycling and end-of-work handling; not done.
On the B200 (fast FP64) the GPU path is the default and should be
measured there.

The helper-thread version was then tried too (one thread per process
outside the teams, taking half of the waiting matches, at least 16, at
most 4096, for a GPU batch; the team threads continuing on the CPU; same
16 workunits): 29.5 s and 679 CPU-seconds against 25.8 s and 474 for the
CPU alone, same polynomials. The batch path costs more CPU than the CPU
path does for the whole job (GPU alone: 668 against 473 CPU-seconds),
since it prepares and optimizes every candidate translation for the
GPU, and the helper is one more thread per process (48 on 32 hardware
threads). A faster GPU does not change that, so it was not kept. The c140
size optimization (457 s here, CPU-bound) stays on the CPU; on the
RTX PRO 6000 it would be about a quarter of the total (estimate).

## Size optimization on the CPU: inline FMA (Nov 2026)

On GPUs with slow FP64 (RTX PRO 4500, RTX PRO 6000) the size optimization
stays on the CPU, where about half its time is real root finding on
double polynomials (`polynomial<double>::roots`: false position with
Horner evaluations through `cado_math_aux::fma`). Without `-mfma` (the
default build, no FP contraction) each `fma` was a call into libm through
the PLT (~9% of the samples). `cado_math_aux::fma` for double and float
now issues `vfmadd213sd/ss` inline when the CPU has FMA (checked once
with `__builtin_cpu_supports`), else `std::fma` as before. A fused
multiply-add is correctly rounded either way, so results are the same:
the c140 work unit (P 125000, 1 ad, `-t 2`) gives the same output, size
optimization 40.2 s -> 37.6 s, wall 44.1 s -> 41.7 s (A/B, 2 runs
each). las's per-special-q norm set-up uses the same code: c140
benchmark relations identical, user CPU ~1% lower (within noise).
