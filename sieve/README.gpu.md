# las on NVIDIA GPUs (branch `cado-nfs-gpu`)

In a CUDA build (see `linalg/bwc/README.gpu.md` for the build), las does
its sieving and its cofactorization on the GPU. Every piece gives the
same relations as the CPU code; each was checked against it.

## What runs where

| Step | GPU code | Used when | Switch |
| --- | --- | --- | --- |
| Sieving: lognorms, bucket and small sieve, survivors, resieving | `las-sieve-gpu*.cpp/.cu` | two sides with factor bases, no sublattices, I <= 2^16, resieving on (all default parameter files) | `CADO_LAS_GPU_SIEVE=0` |
| Top-level bucket fill only | `las-fill-gpu*.cpp/.cu` | when the siever does not apply but the top level is 1 | `CADO_LAS_GPU_FILL=0` |
| Cofactorization (P-1, P+1, ECM chains) | `ecm/facul-gpu*`, `las-cofac-gpu.cpp` | cofactors below 2^126 | `CADO_LAS_GPU_COFAC=0` |

What stays on the CPU: the small sieve set-up per special-q
(`small_sieve_init`), and per survivor the trial division with the primes
the GPU found, the norm checks and the relation output
(`cofactoring_sync`, unchanged).

### The siever

For a special-q, the GPU sieves the whole region, both sides, in slabs of
2^25 positions:

- lognorms: `lognorm_smart` row by row, the same double operations
  (built with `-fmad=false`: the CPU code has no FMA);
- bucket-sieved primes (all levels): the root transform, the
  Franke-Kleinjung reduction and walk of `las-plattice.hpp`, adding logp
  to 16-bit sums; the walks carry over from one slab to the next;
- small primes: the entries that `small_sieve_init` made, in shared
  memory tiles;
- `S = max(lognorm - (sum mod 256), 0)`, survivors as `search_survivors`
  (both sides at most the bound, gcd(i,j) = 1, only (1,0) on row 0);
- resieving: the bucket-sieved lattices are walked again over a bitmap of
  the survivors (a 16 KB summary of it in shared memory), and the small
  entries that `resieve_small_bucket_region` records are tested.

Each bucket region with survivors then goes through
`process_bucket_region_run::run_gpu`: the survivors, and the purged
updates and resieved primes as `purge_buckets` and `resieve` would have
made them, then `cofactoring_sync`.

### Cofactorization

`facul_gpu` runs the method chains of the strategy (the default strategy
or a strategy file) on the GPU, one thread per (candidate, side), with
64- or 128-bit Montgomery arithmetic: stage 1 interprets CADO's own PRAC
bytecode (ECM, P+1) or uses the P-1 exponent, stage 2 runs `stage2.cpp`'s
plan, found factors go through `facul_doit_onefm`'s tests and BPSW. The
curves are computed over Q on the host. Candidates of all threads are
batched (`CADO_LAS_GPU_BATCH`, 65536, or after `CADO_LAS_GPU_WAIT`, 1 s).

## Process layout

las is fastest as one process per GPU with several subjobs:
`-t Machine,16,2` is 16 subjobs of 2 threads, each with its own
special-q. A subjob sieves a special-q on the GPU while its threads
cofactor the previous one (below), and many subjobs keep the GPU busy.
Subjobs of one process share the GPU; separate processes take turns on
it (16 x `las -t 2` on the c120: 4.9 s, one `-t Machine,8,4`: 2.5 s).
NVIDIA MPS should let separate processes share it too (not tested here).

`cado-nfs.py` does this by itself when `linalg/bwc/cuda_devices` finds a
GPU and nothing else is asked: `tasks.sieve.las.threads=Machine,J,2`
with J x 2 = the threads per GPU, and the server hands out at most one
sieving workunit per GPU at a time (`CADO_SIEVE_WU_LIMIT`, in
`scripts/cadofactor/wudb.py`); polynomial selection and the other tasks
keep their clients. With several GPUs, each las claims a GPU at start
(utils/gpu_claim.cpp, lock files in `$TMPDIR/cado-gpu-<uid>`: the las of
a node take the GPUs in turn) and puts it in `CADO_LAS_GPU_DEVICE`, which
the siever, the bucket fill and the cofactorization all read;
`CADO_LAS_GPU_DEVICE` or an MPI/SLURM local rank override the claim.
(Until Nov 2026 las used its own lock, `/tmp/cado-las-gpu-<uid>-<dev>.lock`,
on multi-GPU hosts only, and its parts read the ranks differently.)
Setting `tasks.sieve.las.threads` yourself turns the layout off, except
`gpu-auto`: then each las chooses `Machine,J,2` itself from the CPUs it
may use (its affinity mask) and the GPUs it sees, J = CPUs / GPUs / 2
(at least 2), or `auto` without a GPU, and the rest of the layout stays.
For clusters whose nodes differ (see the top-level README.gpu.md).
`las -t gpu-auto` works outside cado-nfs.py too.

## Checks

- `CADO_LAS_GPU_SIEVE_CHECK=1`: las sieves on the CPU, and
  `process_bucket_region` compares every region's sieve array with the
  GPU's at the positions with gcd(i,j) = 1 (`CADO_LAS_GPU_SIEVE_NORMS=1`:
  the lognorms alone). c120 polynomial: 0 mismatches at I = 12, 13, 14,
  15, special-q 1000000-1000400 and 2000000-2000200. (Before the fix
  below, I = 14 and 15 had one mismatch every few dozen special-q, at
  (1,0) on side 0.)

## Optimizations (Nov 2026), from Nsight Compute

c120 work unit (10000 special-q range, I = 12, `-t Machine,8,4`), las
time 1.49 s -> 1.18 s, relations identical:

- `lsg_resieve_small`: a warp per survivor, its lanes over the small
  entries (it was a thread per survivor: 8% occupancy): 194 -> 25 us.
- `lsg_small_tiles`: tiles of 2^14 positions (32 KB of shared sums: 3
  blocks an SM instead of 1): 242 -> 128 us.
- `lsg_norms`: a thread per row instead of a warp per row (all 32 lanes
  computed the same row: FP64 pipe 72% busy), with 16-byte stores.
- Bug: every small sieve entry hits (0,0), whose 16-bit sum, past 65535
  (I >= 14), carried into (1,0) in the same 32-bit word; (0,0) (never a
  survivor) is no longer sieved.
- Tried and dropped: testing the survivors against each lattice instead
  of walking the lattices again (exact, but slower with the ~5000
  survivors per special-q the GPU keeps), batching the resieve walk,
  larger blocks for it, other subjob layouts, larger mfb/lambda.

cado-nfs.py also gives the GPU siever workunits 5 times longer (each las
pays ~0.3 s to start): c120 sieving 7.5 -> 5.4 min.
- `sieve/ecm/facul_gpu_test` compares the GPU cofactorization with
  `facul_all` on a `las -dump-cofactors` file (identical on 1,268,053
  three-large-prime candidates).
- Relations identical to the CPU in every run below.

## CPU time per special-q (Nov 2026)

With the GPU sieving, the CPU's share is what a faster GPU waits on when
a node has few threads per GPU (a B200 node: ~28 threads per GPU, with a
GPU several times this one). Profile of the c140 benchmark (25000 q,
`Machine,16,2`), CPU samples of las:

| Part | Share |
| --- | ---: |
| writing relations (`get_relation`, `compress`, `operator<<`) | ~23% before, ~6% after |
| norm set-up per special-q (`lognorm_smart`: real roots of polynomials, double) | ~13-17% |
| GPU sieve host side (`las_sieve_gpu_sieve`, launches) | ~11-14% |
| survivors: trial division, norms (`cofactoring_sync`) | ~15% |
| system time: futex (thread pool, GPU waits), per-thread CPU clocks | 4-5 s of ~20 s |

- **Relations are written directly** (`cofac_standalone::format_relation`):
  the primes as 64-bit integers, sorted, printed in hex, instead of a
  `relation` with two `mpz` per prime (the root is not printed),
  `compress()` copying them, and `mpz_get_str`. Taken when no duplicate
  check (`-dup`), no Galois and no descent, all values within 64 bits;
  otherwise the old code. Output byte-identical: the 227,192 relations of
  the benchmark, sorted, equal the old binary's. CPU (user) 16.1-16.4 s
  -> 15.1 s; wall unchanged (GPU-bound here).
- A different allocator (jemalloc, tcmalloc) changed nothing.
- The norm set-up iterates its root finding to the last bit; anything
  faster would change the norms, and so the relations. Left as is.

## Results (one RTX PRO 4500 Blackwell, i9 32 threads)

c120 polynomial and parameters (I = 12, two large primes), special-q
1000000-1008000:

| | wall |
| --- | ---: |
| CPU, `-t 32` | 8.32 s |
| GPU siever, `-t 32` | 3.65 s |
| GPU siever, `-t Machine,4,8` | 1.68 s |
| GPU siever, `-t Machine,8,4` | 1.39 s |

The GPU needs about 2.2 ms per special-q (both sides); with 8 subjobs it
is the limit, at about 7 CPU threads busy. Larger areas: I = 14, 1.77 s
-> 0.63 s; I = 15, 23.5 s -> 1.22 s (8 special-q).

Three large primes (mfb 81/84, lambda 3), one thread: 222 s -> 19 s with
the GPU cofactorization alone (the cofactorization was 92% of it).

### Tried for the large primes, and dropped (Nov 2026)

- Bucket records: lsg_large writing every point (position, lattice) to a
  per-block region, and resieving by a scan of the records instead of a
  second walk. Writing the records made lsg_large 150 -> 277 us and the
  scan took 362 us against 256 for the walk: these kernels are latency
  bound with little work per launch (Nsight Compute: 6% of the issue
  slots used), not bound by the number of points.
- One launch for both sides and both lattice sets (4 times wider):
  resieve kernel time halved, las 3% slower.
- Reusing lsg_large's start states in the resieve (no second
  setup_walk): no change.

### CPU/GPU overlap within a subjob

Without it, a subjob's main thread waits on the GPU while its pool
threads are idle, and then the pool cofactors while the GPU is idle
(each subjob's stream was busy 29% of the time, its threads about 35%).
Now the cofactoring tasks of a special-q sieved on the GPU are not
waited for: the subjob sieves the next special-q meanwhile, in a second
`nfs_work` (the tasks read theirs: configuration, lognorms, per-thread
scratch; they keep their own copy of Q), and a `nfs_work` is reused only
once its tasks are done. Relations are the same (sorted; the order in
the output changes). Off for the descent (each special-q's result is
needed before the next), batch cofactoring, `-sync` and `-exit-early`;
`CADO_LAS_GPU_OVERLAP=0` turns it off.

30000-q c120 range, 4-5 runs each, interleaved:

| Layout | Without | With overlap |
| --- | ---: | ---: |
| Machine,8,4 | 3.11 s | 2.71 s |
| Machine,16,4 | 2.95 s | 2.60 s |
| Machine,16,2 | 2.98 s | **2.45 s** |
| Machine,32,1 / 20,2 | | 2.46 s |

Full c120 (`cado-nfs.py -t 32`): 429 s -> 401 s, sieving 4:07 -> 3:40.

### Slabs sized for the L2 cache; sleeping waits (Nov 2026)

With the overlap above and 16 subjobs, the GPU is the limit: each
special-q's work spent ~11.7 ms on its stream for ~3 ms of kernel time,
queued behind the other subjobs'. A slab was 2^25 positions, so a
subjob's arrays (norms, 16-bit sums and S of both sides, 16 bytes a
position) were 64 MB at I=12 and 512 MB at I=14, against a 64 MB L2
cache shared by all subjobs. Slabs are now 2^22 positions (64 MB per
subjob at most); the walk states carry from one slab to the next as
before. Machine,16,2, relations identical, GPU sieve check 0 mismatches
at I=12, 14, 15 and 16:

| I | 2^25 | 2^23 | 2^22 | 2^21 |
| --- | ---: | ---: | ---: | ---: |
| 12 (30000 q) | 2.42 s | 2.40 s | **2.32 s** | 2.46 s |
| 14 (8000 q) | 13.09 s | 6.63 s | **6.13 s** | 6.62 s |
| 15 (2000 q) | 10.61 s | 5.71 s | **5.41 s** | 6.07 s |

The waits on the GPU now sleep (an event with cudaEventBlockingSync)
instead of spinning in cudaStreamSynchronize, and two of them per
special-q are gone (the copies of the results are ordered on the stream
anyway). The spinning was 3/4 of las's CPU time; las now uses ~1150%
CPU instead of ~1700% for the same wall time. `CADO_LAS_GPU_SPIN=1`
spins.

Full c120 (`cado-nfs.py -t 32`): 401 s -> 385 s, sieving 3:40 -> 3:29.

### Measured limits, and two more designs dropped (Nov 2026)

GPU metrics (nsys, 16 subjobs, I=14): SMs active 94%, warp slots 79%
full, instruction issue 23%, DRAM ~51% of its bandwidth. The GPU is full
of warps waiting on memory. lsg_large fills the memory request queue
(Nsight Compute: lg_throttle and long-scoreboard stalls). Each special-q
at I=14 makes ~59M large-prime hits per side; each costs a global atomic
and a step of the resieve walk. Alone, lsg_large handles ~39G hits/s.

- Norms per 2^26 positions (both sides, one launch) instead of per slab:
  relations identical; I=12 2% faster, I=14/15 6% slower. A per-row
  FP32 compute_y changed nothing, and without the stores the kernel
  takes 121 instead of 172 us: its cost is the sequential row algorithm
  (kept so that the bytes are the CPU's).
- Bucket primes below 2^MID sieved in the shared-memory tiles (one
  modular reduction per prime and row) and resieved along the rows:
  GPU sieve check 0 mismatches and relations identical, but slower at
  every bound, because the work is primes x rows instead of hits (I=14,
  one subjob, GPU time per run: base 0.82 s; MID=15 1.03 s, 16 1.14 s,
  17 1.36 s, 18 1.79 s). The FK walk visits only the hits and is the
  cheaper way.

### The survivors' cofactor checks on the GPU, with CGBN (Nov 2026)

Designed for the production GPUs (RTX PRO 6000 at least, B200): with a
GPU 2.3x or more faster than the RTX PRO 4500, las is CPU-bound, and the
CPU's main work was GMP code on every GPU survivor, 97% of which it then
rejected. That work is now done on the GPU (las-sieve-gpu-cofilter.cu):

- for each survivor and side, the cofactor of cofactoring_sync(): the
  norm |F(i,j)| (lognorm_base::norm, with the fij coefficients) in CGBN
  (256 or 512 bits, chosen from the coefficient sizes), all its 2s, the
  primes of its hits (resieved and bucket primes: the hit lists sorted by
  (position, side, prime) with CUB), the trial division primes (in groups
  whose product is below 2^32: one remainder per group) and, on the
  special-q side of a composite q, its prime factors, each removed fully;
- check_leftover_norm() on it: the mfb, lpb and B^k-gap tests (mpz_get_d
  as truncation), and the base-2 strong probable prime test of
  modredc*_sprp2 (with the one-word variant's shorter search when
  n = +-1 mod 8); above 126 bits, where las uses mpz_probab_prime_p, the
  survivor is left to the CPU, as is a zero norm.

Survivors rejected on a side are dropped before they go back; the CPU
redoes everything for the others. The estimate filter of strategy files
(filter_survivors_by_region) is also applied on the GPU (one 8 KB bit
table of (S0, S1)).

Checks, filter on against off: the survivors entering cofactoring are
the same number (I=12: 323,288; I=14: 387,801; I=15; composite
special-q: 23,432; forced 512-bit kernel: 62,584), relations identical;
175/175 las, sieve, F9, descent and cado-nfs.py tests.

On the RTX PRO 4500 (GPU-bound) it costs GPU time: the kernel takes
1.5 ms per special-q at I=14 alone. 30000-q c120 range, Machine,16,2:

| | CPU off | GPU filter |
| --- | ---: | ---: |
| I=12 wall / CPU | 2.33 s / 28.2 s | 2.76 s / 18.0 s |
| I=14 wall / CPU | 6.07 s / 30.2 s | 6.62 s / 18.6 s |

Full c120: 385 s -> 429 s wall here, total CPU 3739 s -> 2797 s.
CADO_LAS_GPU_COFILTER=0 turns it off (the previous behaviour, faster on
a GPU-bound card), CADO_LAS_GPU_FILTER=0 the estimate filter,
CADO_LAS_GPU_COFILTER_BITS=512 forces the wider kernel (testing).

### Slab size from the GPU's L2 (Nov 2026)

The slab size (2^22 positions, measured best on the RTX PRO 4500's 64 MB
L2, see above) now follows the GPU: one more power of 2 for each doubling
of the L2, rounded, between 2^17 and 2^26. The RTX PRO 6000 (128 MB) and
the B200 (126 MB) get 2^23. **That default is an estimate for those
GPUs, not a measurement**; `CADO_LAS_GPU_SLAB_LOG=<k>` sets 2^k, to tune
it there (relations do not depend on it: identical at 2^21, 2^22, 2^23).
On the RTX PRO 4500 nothing changes (I=14, 4000 q: 3.47 s, previous
build 3.42 s; 2^21 3.62 s, 2^23 3.59 s).

### c140, and the algebraic norms by segment (Nov 2026)

A c140 (two 70-digit primes, cado-nfs.py -t 32, RTX PRO 4500): 5725 s,
of which sieving 4674 s (81.6%; c120: 62.7%), polynomial selection 529 s,
filtering 239 s, linear algebra 223 s, sqrt 43 s. Sieving grows 17.7x
from the c120, the other stages 5-9x. Its las is GPU-bound: 133% CPU
(19.8 CPU-s in 14.9 s per workunit), 71 ms of GPU stream time per
special-q; the cofactor filter rejects 97.3% of the survivors (so the CPU
has little to do). Kernel time at I=13: norms 32%, lsg_large 23%,
lsg_resieve_large 20%, small tiles 13%, cofactor filter 6%.

- The "rejected by the GPU cofactor filter" count of the report was 0
  since the GPU drops the rejected survivors itself (cf21ab9ba); it now
  comes from the GPU (lsg_out.nrejected).
- The algebraic side's norms (degree > 1) now run a thread per (segment
  of the G approximation, row), lsg_norms_alg: fill_rat_row does each
  segment from scratch, so the bytes are the same (GPU sieve check 0
  mismatches at I=12, 13, 14; c140 relations identical). That launch
  went 551 -> 292 us (the rational one: 276 us), all norms 3321 -> 2317
  ms, but the GPU stream time only 36.33 -> 35.95 s and the wall time
  2.75 -> 2.73 s: the norms kernel has few threads, and the other
  subjobs' kernels already filled the GPU while it ran.
  CADO_LAS_GPU_NORMS_ROW=1 uses the row kernel.

### Slab size from the factor base too (Nov 2026)

Each slab reads and writes every lattice's walk state, so the slab that
balances that traffic against the L2 grows with the number of
bucket-sieved entries. c140 parameters (I=13, 1.63 million entries on the
two sides): 2^21 10.43 s, 2^22 7.62 s, **2^23 6.75 s**, 2^24 8.08 s, 2^25
11.38 s (25000 q, Machine,16,2, relations identical); c120 (426,000
entries): 2^22 best. The default adds half a power of 2 per doubling of
the entries over 426,000 (rounded) to the L2-based value: c120 2^22,
c140 2^23. `CADO_LAS_GPU_SLAB_LOG` still sets it outright.

### The walk state split: positions and steps (Nov 2026)

lsg_large and lsg_resieve_large read and wrote each lattice's whole
32-byte walk state for every slab (with the lattice: about 136 bytes per
lattice and slab over the two passes), although only the position
changes, and a lattice often has no point in a slab (c140: 1.6 million
lattices; the largest primes hit a 2^23-position slab about once). The
steps (24 bytes) are now written once per special-q, the positions in
two 8-byte arrays (this slab's, the next one's, swapped after each
slab): a lattice with no point in the slab costs reading its position
(sieving) and copying it (resieving), 24 bytes. The resieving pass
writes the next slab's positions (from this slab's, so that a redo after
an overflow starts from the same place); without it (check mode)
lsg_large does. GPU sieve check 0 mismatches at I=12..15; relations
identical. Machine,16,2: c140 (25000 q) 6.76 -> 5.84 s (-13.7%), c120
I=12 (30000 q) 2.75 -> 2.65 s. The slab default (2^23 at c140, 2^22 at
c120) is still the best or within 2%.

### Resieving a slab and sieving the next in one pass (Nov 2026)

At c140 a lattice has about three points a slab, and lsg_resieve_large
(210 us a call) cost nearly what lsg_large did (249 us): both were mostly
the work per lattice. lsg_resieve_sieve resieves slab s and goes on to
sieve slab s+1 into its cleared sums, so that per-lattice work is done
once a slab; it writes the position at the start of slab s+1, which the
next pass resieves from, and leaves its input alone, so an overflow redo
(the sums cleared again) is exact. The first slab is still sieved by
lsg_large, the last resieved by lsg_resieve_large; the check mode (no
resieving) is unchanged. Relations identical. Machine,16,2: c140 5.71 ->
5.63 s, c120 I=12 2.63 -> 2.59 s, I=14 (32 slabs) 6.04 -> 5.59 s.
CADO_LAS_GPU_FUSE=0 keeps the separate passes.

### The cofactor filter's prime test in 64-bit arithmetic (Nov 2026)

The base-2 strong probable prime test (CGBN modular_power at 256 bits)
was 80% of lsg_cofilter_kernel at c140 (752 us a call, 154 without it):
a few survivors reaching it set the kernel's length, and the cofactors
there have at most mfb bits (58 at c140). Up to 64 bits it now runs in
64-bit Montgomery arithmetic (sprp2_u64, the same test as
modredcul_sprp2, with its shorter search when n = +-1 mod 8); from 65 to
126 bits the CGBN version stays. Checked against a Python version of
mod_ul_common.c's mod_sprp2 on 302,019 values (all odd n < 200,002, the
small base-2 strong pseudoprimes and 3825123056546413051, n near 2^63
and 2^64, 200,000 random 64-bit values): 0 differences; survivors
entering cofactoring identical with the filter on and off (c140 65,312,
c120 108,151), relations identical. The kernel: 752 -> 155 us; las
(Machine,16,2): c140 5.54 -> 5.45 s, c120 2.62 -> 2.37 s. Removing a
prime now uses the remainder that cgbn_div_ui32 returns (one call a
factor and one more, instead of remainder, division, remainder).

### Survivor search and small-prime resieving: fewer threads (Nov 2026)

Weighted by threads launched times time, lsg_survivors (a thread per
position, two bytes read, for about a thousand survivors a slab) and
lsg_resieve_small (a warp per survivor slot of the list's capacity, since
the count is only on the device) were the top two at c140 (41%, 21%).
lsg_survivors now takes 16 positions a thread (16-byte loads, compared 4
bytes at a time with __vcmpleu4), lsg_resieve_small runs a fixed grid
(8 blocks an SM) whose warps go over the survivors, with a 32-bit
remainder when r j + I/2 fits. Relations and survivors entering
cofactoring identical. The weighting overstated them (most of those
threads ended at once): c120 2.35 -> 2.28 s, c140 5.46 -> 5.44 s.
