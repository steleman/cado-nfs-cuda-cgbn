# GPU support: hardware and builds

The GPU code of this tree (polynomial selection, the las siever and its
cofactorization, characters, bwc, sqrt) is meant for production GPUs:

- minimal expected configuration: **NVIDIA RTX PRO 6000 Blackwell**
  (sm_120, 96 GB, 188 SMs, 128 MB L2);
- optimal configuration: **NVIDIA B200** (sm_100) or better.

Development and every measurement in the README.gpu.md files of the
subdirectories were done on an RTX PRO 4500 Blackwell (sm_120, 32 GB,
82 SMs, 64 MB L2) with an i9-13900K, which is not a production
configuration. Some changes are slower on that card by design: it is
GPU-bound, while the production GPUs are 2.3 times (or more) faster, so
that moving CPU work to the GPU pays off there (for example the GPU
cofactor filter of las, sieve/README.gpu.md). **Any figure given for the
RTX PRO 6000 or the B200 is an estimate; none has been measured yet.**

### CGBN

You will need the [CGBN fork from my Github repo](https://github.com/steleman/nvidia-cgbn)
to build cado-nfs with CGBN and GPU / CUDA support. The original [NVIDIA CGBN](https://github.com/NVlabs/CGBN) only supports unsigned arbitrary precision integers and has no
support for signed arbitrary precision integers.

### Documentation
Additional documentation about the design and implementation of this fork, as well as
compile-line flags and environment variables knobs can be found in [doc-gpu](doc-gpu).

## Building for the production GPUs

`CMAKE_CUDA_ARCHITECTURES` decides which GPUs the binaries can run on.
If it is not given:

1. `$CADO_CUDA_ARCHITECTURES`, if set (for example
   `CADO_CUDA_ARCHITECTURES="100-real;120-real;120-virtual"`);
2. else the GPUs of the build host (`native`), when `nvidia-smi` finds
   one;
3. else (a build host without a GPU, such as a cluster's login node)
   `100-real;120-real;120-virtual`: B200, RTX PRO 6000, and PTX that
   later GPUs can compile at load time.

A binary built `native` on one GPU and run on another kind (built on an
RTX PRO 6000, run on a B200) has no code for it. Each GPU module then
says so once and uses its CPU path (las: siever, bucket fill,
cofactorization; polyselect: collisions, size and root optimization;
characters; sqrt), with the same results; bwc with `mm_impl=cuda` stops
with a message, since it has no CPU path at that point. For a cluster,
build with the list above.

All of the CUDA code builds for sm_100 (checked with CUDA 13.2,
`100-real;120-real;120-virtual`, no errors); the sm_120 code of that
build gives the same relations as the `native` one. It has not run on
an sm_100 GPU yet.

CGBN (header only) is looked up in `$CGBN_HOME/include`, then
`/usr/local/include` and `/usr/include`.

Run-time choices follow the device, not the development card: SM counts
are read from the device (las persistent kernels, bwc), the las slab size
follows the L2 size (an estimate for the production GPUs,
`CADO_LAS_GPU_SLAB_LOG` overrides it), and the GPU size
optimization of polyselect is used when the FP32/FP64 throughput ratio
is at most 4 (B200: yes; RTX PRO 6000 and 4500, ratio 64: the CPU path).

## Several GPUs on a node

A B200 node has 8 GPUs. cado-nfs.py runs up to one sieving workunit per
GPU at a time (sieve/README.gpu.md), and several polyselect clients. A
GPU program that is not told which GPU to use (`CADO_LAS_GPU_DEVICE`,
`CADO_POLYSELECT_GPU_DEVICE`, `CADO_SQRT_GPU_DEVICE`,
`CADO_CHARACTERS_GPU_DEVICE`, or an MPI / Slurm local rank) claims one
(utils/gpu_claim.cpp): the processes of a program take the GPUs in turn
through lock files in `$TMPDIR/cado-gpu-<uid>` (or `/tmp`), slot 0 of
each GPU first, then slot 1, and so on; a lock goes away with its
process. Before this, every las used GPU 0 and polyselect / sqrt spread
by pid. All the GPU contexts of one process use the GPU it claimed.
`CUDA_VISIBLE_DEVICES` still restricts the GPUs seen.

Tested on the single-GPU development box with `CADO_GPU_SIMULATE_NDEV=4`
(`CADO_GPU_CLAIM_VERBOSE=1` prints each claim): 8 las at once took
simulated GPUs 0, 1, 2, 3 twice each; a las started after they exited
took GPU 0 again, one started while another ran took the next GPU. A
real multi-GPU node has not been tested.

## Clusters: several GPU nodes (Nov 2026)

The sieving and polynomial selection run as workunits of cado-nfs.py's
server, done by cado-nfs-client.py processes on any number of nodes:

    cado-nfs.py --server -t <threads of a sieving node> <N> server.whitelist=<nodes' network>
    # on each node, several clients (the server prints --server and --certsha1):
    cado-nfs-client.py --server=https://<server>:<port> --certsha1=<sha1>

- With the GPU siever, each las takes a GPU's share of the node
  (`tasks.sieve.las.threads`, from `-t` and the number of GPUs, see
  sieve/README.gpu.md), so the server hands out at most one sieving
  workunit per GPU at a time. **This limit is per node**: each client
  sends its host name and its node's number of GPUs with its requests
  (`nvidia-smi -L`, after `CUDA_VISIBLE_DEVICES`; `CADO_CLIENT_NGPU`
  overrides it), and a node gets sieving workunits while its clients hold
  fewer than that many. Before, the limit was the server's own number of
  GPUs for the whole cluster. Older clients (no host sent) still count
  against that global limit.
- On the node, each las claims a GPU of its own (lock files, see
  "Several GPUs on a node"), as do polyselect, sqrt and characters.
- The server decides `tasks.sieve.las.threads` from its own GPUs. When
  the server node has no GPU, or not as many as the sieving nodes, set
  `CADO_LAS_NGPU=<GPUs of a sieving node>` for cado-nfs.py, and `-t` to a
  sieving node's threads. That fixed layout assumes sieving nodes of one
  kind. For mixed nodes, give `tasks.sieve.las.threads=gpu-auto`: each
  las then works out its share of its own node (the CPUs it may use,
  from its affinity mask, so taskset and Slurm limits count, divided by
  the GPUs it sees, in subjobs of 2 threads: `Machine,16,2` for 32 CPUs
  and 1 GPU, `Machine,4,2` under `taskset -c 0-7`; plain `auto` without
  a GPU). The rest of the GPU layout (one sieving workunit per GPU of a
  node, the larger q-ranges) applies as with the fixed one. Checked: the
  same relations with gpu-auto and a fixed layout, and a c60 with
  `tasks.sieve.las.threads=gpu-auto` (every las got `-t gpu-auto`,
  factors correct).
- Linear algebra on several nodes: build with MPI (configure with
  `MPI=1` in the environment; note that a later automatic reconfigure
  without it turns MPI off again) and give the nodes:
  `tasks.linalg.bwc.hosts=node1,node2,...` (or
  `tasks.linalg.bwc.hostfile=<file>`, or run inside a Slurm allocation).
  bwc.pl then runs one MPI job per GPU of every node (`mpi = nodes x
  GPUs per node`, `thr=1`; Open MPI with `--map-by node`, each job on the
  GPU of its local rank, vectors exchanged with NCCL). The GPUs per node
  are the local node's, or `CADO_BWC_GPUS_PER_NODE` when bwc.pl starts on
  a node without (or with other) GPUs, such as the server. An explicit
  `tasks.linalg.bwc.mpi` is kept. Nodes need passwordless ssh between
  them (Open MPI starts its daemons that way) and, for NCCL, a network it
  can use (`NCCL_SOCKET_IFNAME`, InfiniBand). Running bwc.pl directly,
  `mpi_extra_args="-x NAME"` passes other variables to the jobs (cado-nfs.py
  has no parameter for it, nor for bwc.pl's `gpu=` and `gpu_block=`: export
  such variables in the environment Open MPI forwards, or run bwc.pl by
  hand). Filtering, characters and sqrt run on the server node.

  Tested on one machine only: `--show` gives `mpi=8 thr=1` and
  `mpiexec -machinefile <2 hosts> --map-by node -n 8` for two nodes of 4
  GPUs; the multi-job GPU path (mpi=2 on 2 simulated GPUs, NCCL) gives the
  kernel in 23 s on the c75 matrix. Two node names for this machine make
  Open MPI start the second over ssh, which has no keys here (a CPU-only
  run fails the same way), so a real multi-node run is still to be done.

Tested on one machine: a c75 with `--server` and four clients posing as
two nodes of 1 GPU each (`CADO_CLIENT_HOST`, `CADO_CLIENT_NGPU`, test
only): two sieving workunits at a time, at most one per "node" (the old
limit allowed one in all), factors correct; the server's assignment
checked directly with a node of 2 GPUs, one of 1 and an old client. Not
tested on real nodes.
