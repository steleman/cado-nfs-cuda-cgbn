#ifndef CADO_UTILS_GPU_CLAIM_H
#define CADO_UTILS_GPU_CLAIM_H

/* Which GPU a process should use, when nothing says (no CADO_*_DEVICE,
 * no MPI or Slurm local rank): the processes of a node take the GPUs in
 * turn, by claiming lock files (slot 0 of GPU 0, 1, ..., then slot 1 of
 * each, ...). On a node with 8 GPUs, cado-nfs.py runs up to 8 las at a
 * time; without this, all of them used GPU 0. A lock goes away with its
 * process, so the next one takes its GPU.
 *
 * tag: the program ("las", "polyselect", ...); every call with the same
 * tag in a process gets the same answer. Returns a number in [0, ndev).
 * The locks are in $TMPDIR (or /tmp)/cado-gpu-<uid>.
 * CADO_GPU_SIMULATE_NDEV=n claims among n GPUs (testing: the result is
 * still taken mod ndev) and CADO_GPU_CLAIM_VERBOSE=1 prints the claim. */

#ifdef __cplusplus
extern "C" {
#endif

int cado_gpu_claim(const char * tag, int ndev);

#ifdef __cplusplus
}
#endif

#endif
