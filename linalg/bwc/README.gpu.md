# bwc on NVIDIA GPUs (GF(2), branch `cado-nfs-gpu`)

The block Wiedemann linear algebra for factoring (`bwc.pl`, GF(2)) runs its
sparse matrix products on the GPU. lingen stays on the CPU.

## Build

CUDA is detected at configure time (`ENABLE_CUDA`, on by default when `nvcc`
is found; `CUDACXX=/usr/local/cuda/bin/nvcc` if it is not on the `PATH`).
`CMAKE_CUDA_ARCHITECTURES` defaults to `$CADO_CUDA_ARCHITECTURES`, else
the GPUs of the build host (`native`), else (no GPU on the build host)
`100-real;120-real;120-virtual`; see the top-level README.gpu.md. Set it
explicitly when building for other machines, for example `120` for
Blackwell RTX PRO cards or `100` for B200. NCCL is used when found
(`ENABLE_NCCL`, `NCCL_HOME`).

For several GPUs, build with MPI (`MPI=1` with `mpicc` on the `PATH`; with
CMake 3.31, `MPI=<path>` makes `find_package` look for a config package and
fail with Open MPI).

## Use

Nothing to change: in a CUDA build with a GPU present, `bwc.pl` (and so
`cado-nfs.py`) uses it (`gpu=auto`):

- `mm_impl=cuda`, one GPU per matrix block;
- one GPU: `thr=1`; several GPUs in an MPI build: one MPI job per GPU
  (`mpi=<ngpu>`, `thr=1`), vectors exchanged GPU to GPU through NCCL;
  several GPUs without MPI: one thread per GPU, vectors on the host;
- `lingen_thr=` the `thr=` that was asked for (lingen is OpenMP-parallel; it
  used to run on one thread by default);
- the default block size `m=n=64` becomes `m=n=simd=128` (`gpu_block`): a
  128-bit product costs the GPU about what a 64-bit one costs, so krylov and
  mksol need half as many.

`gpu=0` keeps the CPU code, `gpu=1` fails without a GPU, `gpu_block=64` keeps
`m=n=64`. With `mpi=` given by hand, each job gets one GPU (by local rank,
`CADO_BWC_CUDA_DEVICE` forces one).

Environment switches:

| Variable | Effect |
| --- | --- |
| `CADO_BWC_CUDA_RESIDENT=0` | vectors on the host (copies around each product) |
| `CADO_BWC_GPU_COMM=nccl\|mpi` | multi-GPU transport (default NCCL if built in) |
| `CADO_BWC_CUDA_VERBOSE=1` | per-process product count and kernel/copy times |
| `CADO_BWC_SIMULATE_GPUS=<n>` | testing: pretend there are n GPUs (they share the real ones; pass it to the ranks with `mpi_extra_args="-x CADO_BWC_SIMULATE_GPUS"`) |

## How it works

- `matmul-cuda.cpp`, `matmul-cuda-kernels.cu`: the `cuda` matmul backend. The
  block is uploaded in the `matrix_u32` layout, cut into work items of at most
  4096 indices (rows are very uneven: median 33, longest 182,916 on the c120
  block), one warp per item, XOR of the gathered source elements, atomic XOR
  for the pieces of a split row. The other direction (secure) is an atomic
  scatter.
- `mmt_cuda` (`matmul_top`): krylov, mksol and secure keep their vectors on
  the GPU for a whole interval (x dot products, `addmul_tiny` and the product
  run there); the host sees them only at checkpoints. With several jobs, a
  product is followed by an XOR reduce-scatter (NCCL has no XOR reduction:
  grouped send/recv plus an XOR kernel) and an in-place all-gather, the same
  data movement as `mmt_vec_reduce` + `mmt_vec_broadcast`.
- NVSHMEM is not used: the only per-iteration traffic is the bulk vector
  exchange, which NCCL does at link speed, and the dot products are summed
  once per interval.

## Results (one RTX PRO 4500 Blackwell, i9 32 threads)

c120 matrix, 635,213 rows, 106.8 entries per row:

| | CPU (bucket, 32 threads) | GPU |
| --- | ---: | ---: |
| prep | 2.6 s | 2.7 s |
| secure | 20.2 s | 4.6 s |
| krylov | 152.2 s | 11.8 s |
| lingen | 24.5 s (1 thread) | 13.1 s (32 threads, m=n=128) |
| mksol | 76 s | 6.6 s |
| bwc total | 276 s | 40.3 s |

Per product: 1.02 ms at 64 bits, 1.04 ms at 128, 1.3 ms at 256 (bench). The
whole `cado-nfs.py` c120 went from 28.6 to 25.4 min; sieving is now most of
it.

Checks: V, A, S and C files byte-identical between the host loop and the GPU
loop, and between 1x1 and 2x2 / 3x1 / 1x3 MPI grids (both transports, 4
ranks on one GPU); full runs factor the c120. Multi-GPU speed is untested
(single-GPU host).

## lingen on the GPU: measured, not done (Nov 2026)

On the c120 (m = n = 128, 10000 iterations), lingen_b64 takes 12.3 s
wall (106 s CPU): 6.8 s in the basecase (128 sequential calls on length
79, which a GPU cannot run in parallel) and 5.4 s in the recursive
products, plain gf2x_mul per entry (all below the ternary/cantor FFT
thresholds of 6400). The GPU has no carry-less multiply; its 1-bit
tensor-core MMA (mma.sync m16n8k256 b1 and.popc, correct on sm_120)
peaks at ~122 T GF(2) multiply-adds/s on the RTX PRO 4500, against
~92 T achieved by gf2x with PCLMUL on 32 threads (~340 T peak). A GPU
lingen would at best match the CPU here, so it was not written.
recursive:64 instead of 128 gave the same F files in 10.8 s instead of
11.8 s with 35% more CPU; not changed.

## The product kernel: measured limits (c140, Nov 2026)

c140 matrix: 1,679,568 rows, 210.3 M entries (125 per row), m = n = 128
(one 128-bit vector, `W = 2`): 3.34-3.44 ms per product, so krylov 99.6 s
and mksol 48.3 s of the 221 s of linear algebra.

- The gathers set the speed: 210 M random 16-byte reads from a 27 MB
  vector (in L2) per product, ~2 TB/s of L2 sector traffic; the index
  stream is only ~250 GB/s of DRAM.
- Fewer lanes per row (16, 8 or 4 lanes per row, 2-8 rows per warp,
  more loads in flight per lane, fewer shuffle steps; split rows still a
  warp each): 3.36 / 3.44 / 3.60 ms against 3.36-3.44 ms for a warp per
  row; output identical. Not kept.
- No hot columns to keep in shared memory: after balancing, the source
  indices are uniform (each power-of-two range holds its share), and the
  heaviest 8192 columns hold 1.7% of the entries (65536: 10.5%).
- Wider vectors (`gpu_block=256`, m = n = simd = 256), measured on the
  c140 with cado-nfs.py's bwc.pl command otherwise unchanged:

  | | m = n = 128 (default) | m = n = 256 |
  | --- | ---: | ---: |
  | per product | 3.47 ms | 5.02 ms (1.45x) |
  | prep + secure | 29.8 s | 30.0 s |
  | krylov | 99.6 s | 83.1 s |
  | lingen_b64 (32 threads) | 41.1 s | 78.0 s |
  | mksol | 48.3 s | 38.3 s |
  | gather | 1.6 s | 2.5 s |
  | total | 220.4 s | 231.9 s |

  Both find left nullspace vectors. The products cost 1.45x per product
  on this matrix (1.25x on the c120's smaller one), so krylov and mksol
  gain only 0.79-0.83x, and lingen nearly doubles: 128 stays the
  default. 256 would pay only with a GPU lingen or a matrix whose
  products dominate more.
- Counters (Nsight Compute) need root here (`RmProfilingAdminOnly`);
  these are timings.

On a PRO 6000 (128 MB L2, more bandwidth: untested estimate) the vector
of matrices up to twice this size still fits in L2.
