# Strategy files: writing, testing, improving

`las -file-cofact X.st` replaces the default cofactorization strategy by a
table: for each pair of cofactor sizes, which methods to run, on which
side, in which order. This file says how to write one and how to tell
whether it is better than what you had. The rest of this directory (see
README) is a generator; nothing below requires it.

**TL;DR**: the commands that train a strategy file for a parameter set,
and what they gave on RSA-240, are in
[section 8](#8-tldr-train-a-strategy-for-a-parameter-set).

## 1. The format

    # comments anywhere; newlines are not significant
    d = {
        PM1 315 2205
        PP1-27 525 3255
        ECM-TM12(2) 105 3255
        ECM-TM12(1) 315 5355
    }
    94 64: { 0: d[:3] 1: d 0: d[3:] }
    95-97 30-37: { 1: PM1 1200 16000 ECM-TM12 200 12000 STOP MPQS }

- `name = { CHAIN }` defines a fragment. `R0 R1: { CHAIN }` gives the
  chain of a cell: R is the bit length of a cofactor left after trial
  division, or a range `a-b` of them. A later cell overrides an earlier
  one, and an empty chain rejects the cell.
- A method is `NAME B1 B2`, NAME one of `PM1`, `PP1-27`, `PP1-65`,
  `ECM-M12`, `ECM-M16`, `ECM-B12`, `ECM-TM12`, optionally with a curve
  parameter (`ECM-B12(11)`); a bare ECM name takes the next parameter of
  the default sequence. `MPQS` takes no bounds.
- `0:` and `1:` set the side of the methods that follow (0 at first).
  `name` inserts a fragment, `name[a:b]` its methods a to b-1, as in
  Python. Side switches inside a fragment stay inside it: a fragment
  without one serves either side.
- Methods run in the order written, the two sides interleaved. A side
  gives up after its last method, and a side that gives up loses the
  pair: the side most likely to fail should come first.
- `STOP` after a method: if that side has not split by then, give up.
  The methods after it only finish the composites left after a split.
- A cell that is not in the file rejects the pair, unless neither side
  needs factoring. A file that leaves cells out also drops, right after
  the sieve, the survivors whose estimated sizes are not within 6 bits
  below / 4 bits above an accepted cell.

The tools write this syntax, with fragments for the chains that recur
and ranges for the cells that share one.

mfb0/mfb1 still bound the cells that can be reached, and lambda0/lambda1
the number of survivors. `-adjust-strategy` is unrelated (it is about the
shape of the sieve region).

## 2. A starting point

- The default strategy, written out cell by cell, behaves as no file at
  all; `facul_strategy_oneside::default_strategy(n)` gives its chain with
  n curves. Drop cells or shorten chains from there.
- An existing table, translated (for instance one from another siever;
  mind the order of the sides in its parameter files).
- A per-cell choice of prefixes of the default chain (Kleinjung's rule,
  section 5), from a calibration dump (section 3):

      cofactoring_trajectories -in cal.dump -out traj -times times \
          -lim0 ... -lpb0 ... -mfb0 ... (same for side 1) -ncurves 40
      strategies_from_trajectories -traj traj -times times -C0 $C0 -out X.st

  This works where the default chain is the right family of methods: up
  to three large primes and 126-bit cofactors, say. In the
  more demanding four large primes setting, a few curves with B1 from 200
  to 1500, a `STOP` and a quadratic sieve do better.
  The test `F9_strategies_pipeline` runs this chain.

## 3. A calibration sample

1. Special-q in the q range the file is for, and *not* the ones you will
   evaluate on (per-cell choices overfit their own sample).
2. A strategy file that accepts every cell with one cheap method per side,
   for instance `[S0: PM1, 30, 100 ] [S1: PM1, 30, 100 ]` for all r0 <=
   mfb0, r1 <= mfb1. Not an empty file: it would drop nearly all survivors
   before they are seen.
3. mfb and lambda as wide as anything you mean to try.
4. Two las runs on the same special-q:
   - with `-dump-cofactors cal.dump`, which writes the pairs entering the
     cofactorization (one `a b N0 N1` line each). This serializes las on
     its output lock: do not time this run.
   - with `-T`, for C0, the cost of producing the pairs: total cpu time
     minus the last term of `factor (... + ... + X)`.

A calibration sample can take multiple hours to create.

## 4. Replaying offline

    replay_cofactoring -in cal.dump -lim0 ... -lim1 ... -lpb0 ... -lpb1 ... \
        -mfb0 ... -mfb1 ... -file-cofact X.st -t $(nproc) \
        -cells X.cells -smooth X.smooth

prints the pairs, the smooth ones, and the cpu time. `-cells` writes
`r0 r1 calls smooth time` (cpu seconds) per cell, `-smooth` the indices of the
smooth pairs (to compare two files). It runs `facul_all()` as las does:
same relations, and cpu times within a few percent of las's. Use all
hardware threads: at full load, timings repeat to 1-2%; at half load they
drifted by 10%. Replaying the first lines of a dump is also the quickest
syntax check of a file.

## 5. Improving the file

The criterion is Kleinjung's: with T the cpu time of the whole run and Y
its relations, a relation is worth T/Y; a cell, or a method in a cell, is
worth keeping if it brings relations for less.

- **Side order.** Which pairs are smooth does not depend on the order of
  the methods in a cell, only the time does. Cut each side's chain into a
  head (up to `STOP`, or before the first `MPQS`) and a tail, replay the
  six orderings (h0 h1 t0 t1, h1 h0 t1 t0, c0 c1, ...), and keep the
  fastest one per cell. This can reduce timing by about a third on large
  experiments. `refine_strategy_file -reorder` does this.
- **Cells that do not pay.** Drop a cell when, over its 5x5 neighbourhood
  (single cells are too noisy), the replay spends more than 1.4 T/Y per
  smooth pair. It can be worth a few % in cpu time per relation, at a
  very marginal cost in yield. `refine_strategy_file -prune 1.4 -C0 $C0`
  does this.

      refine_strategy_file -in cal.dump -lim0 ... (as replay_cofactoring) \
          -file-cofact X.st -out Y.st -reorder -prune 1.4 -C0 $C0 -t $(nproc)

  Reorder before pruning: the pruning uses the times of the reordered
  file. The test `F9_strategies_pipeline` runs both.
- **Methods.** ECM-TM12 has precomputed chains for B1 = 105, 115, ...,
  1021, 256, 315, 512. `MPQS` is `mpqs_doit()`, which is slow.  And
  cofactors above 126 bits go through the mpz layer (an ECM curve costs
  about 10 times what it does at 126 bits). This calls for code
  improvements.

## 6. Testing in las

- The same special-q for both files (a `-todo` file, or `-q0 -q1
  -random-sample N -seed S`), none from the calibration, on the same
  node, one run after the other.
- Compare relations, cpu-s per relation (blind to the tail of the run),
  wall-clock times threads per relation, and the `factor` cpu.
- Several q slices; every machine type you will run on (relative costs
  differ: on Skylake-SP the cofactorization weighs twice what it does on
  Zen 4c, and the best pruning threshold moves).
- Noise: about 1% in relations between seeds, a few percent in wall time
  between nodes.

## 7. Are the new relations worth it?

More relations per second is not the goal. A file that accepts other
cells changes which relations you get, and the matrix with them: check
with `scripts/estimate_matsize/estimate_matsize.sh` on the same q range.
On RSA-240, three large primes on side 0 with a generated file gave 3%
more relations and a 1.65% smaller matrix. On DLP-240, `-batchmfb1 105`
gave 22% more relations and a matrix 31% larger at equal relation count.

## 8. TL;DR: train a strategy for a parameter set

The whole sequence, on the two-sided part of the RSA-240 record (q in
[0.8e9, 2.1e9]), with three large primes on side 0 for the trained
strategy (`WIDE`). `$DATA` is a directory containing (at least) two
files, which are `rsa240.poly` and `rsa240.fb1.zst`, from the RSA-240
README of the cado-nfs records repository; `$CADO` is a directory with a
cado-nfs checkout (so that
`$CADO/sieve/strategies/README.strategy-files.md` is the file you're
currently reading). The result is put in a subdirectory of the current
directory called `strat.<something random>/`

    eval `cd $CADO ; make show`
    # if rsa240.fb1.zst is missing
    # fb_args=(-poly $DATA/rsa240.poly -side 1 -lim 21e8 -maxbits 16)
    # $build_tree/sieve/makefb "${fb_args[@]}" -t 16 -out $DATA/rsa240.fb1.zst
    LAS=(-poly $DATA/rsa240.poly
         -fb1 $DATA/rsa240.fb1.zst
         -fbc /tmp/rsa240.fbc
         -lim0 18e8 -lpb0 36 -mfb0 72  -lambda0 2.2
         -lim1 21e8 -lpb1 37 -mfb1 111 -lambda1 3.2
         -q0 8e8 -q1 2.1e9 -sqside 1
         -A 32 -adjust-strategy 2
         -bkmult 1,1l:1.15,1s:1.4,2s:1.1
         -bkthresh1 90000000
         -t auto)
    WIDE=(-mfb0 108 -lambda0 3.0)
    SIZES=(-lim0 18e8 -lim1 21e8 -lpb0 36 -lpb1 37 -mfb0 108 -mfb1 111)

    RES=$(mktemp -d strat.XXXXXXXXX)
    # 1. calibration: every cell accepted, one cheap method per side
    echo "0-108 0-111: { 0: PM1 30 100 1: PM1 30 100 }" > $RES/all.st
    $build_tree/sieve/las "${LAS[@]}" "${WIDE[@]}" -random-sample 64 -seed 1 \
        -file-cofact $RES/all.st   \
        -dump-cofactors $RES/cal.dump  \
        -out $RES/cal-dump.out
    $build_tree/sieve/las "${LAS[@]}" "${WIDE[@]}" -random-sample 64 -seed 1 \
        -file-cofact $RES/all.st   \
        -T  \
        -out $RES/cal.out
    # C0: total cpu time minus the last term of "factor (... + ... + X)"
    C0=$(perl -ne '/^# Total cpu time ([\d\.]+)s.*factor [\d\.]* \([\d\.\+]+ \+ ([\d\.]+)\)/ && print $1 - $2, "\n";' $RES/cal.out)
    # 2. what the default chain does on each side of each pair
    $build_tree/sieve/strategies/cofactoring_trajectories   \
        -in $RES/cal.dump  \
        -out $RES/traj \
        -times $RES/times \
        "${SIZES[@]}"   \
        -ncurves 40 -t $(nproc)
    # 3. per cell, prefixes of the default chain and their interleaving
    $build_tree/sieve/strategies/strategies_from_trajectories   \
        -traj $RES/traj    \
        -times $RES/times  \
        -C0 $C0 \
        -out $RES/gen.st
    # 4. side order per cell, then pruning
    $build_tree/sieve/strategies/refine_strategy_file   \
        -in $RES/cal.dump  \
        "${SIZES[@]}"   \
        -file-cofact $RES/gen.st \
        -out $RES/final.st \
        -reorder -prune 1.4 -C0 $C0 -t $(nproc)
    # 5. other special-q, against the default strategy
    $build_tree/sieve/las "${LAS[@]}" -random-sample 256 -seed 2 -T -out $RES/default.out
    $build_tree/sieve/las "${LAS[@]}" "${WIDE[@]}" -random-sample 256 -seed 2 -T \
        -file-cofact $RES/final.st -out $RES/final.out
    grep -c '^[^#]' $RES/default.out $RES/final.out
    grep '^# Total cpu time' $RES/default.out $RES/final.out

On one 64-thread Xeon Gold 6130 node: calibration 7 minutes (6.4M
pairs, 544 MB), steps 2 to 4 11 minutes, evaluation 13 minutes per run.
The two changes, the strategy file and the wider box, separated (same
node, same 256 special-q):

| box | strategy | relations | cpu-s | relations per kcpu-s | cofactorization cpu-s |
|---|---|---|---|---|---|
| record | default | 14369 | 50183 | 286.3 | 3622 |
| record | trained for the record box | 14107 | 47869 | 294.7 (+2.9%) | 1643 |
| `WIDE` | default | 15900 | 109518 | 145.2 (-49%) | 60571 |
| `WIDE` | trained for `WIDE` | 15053 | 48457 | 310.6 (+8.5%) | 1958 |

The wider box only pays with a strategy that picks its cells: the
trained file keeps 95% of the relations that the wider box can find, for
44% of its cpu time. Step 4 adds nothing to a generated file (its cells
are chosen by the same rule, its interleaving is already the cheapest);
it matters for translated or hand-written files.
