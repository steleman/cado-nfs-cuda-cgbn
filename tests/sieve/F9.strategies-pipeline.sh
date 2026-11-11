#!/usr/bin/env bash

# The pipeline that builds strategy files from what the default chain
# does on real cofactors:
#   las -dump-cofactors -> cofactoring_trajectories
#       -> strategies_from_trajectories -> strategy file
# Which pairs a strategy built from prefixes of the default chain finds
# does not depend on timings: the trajectories say it exactly, and
# strategies_from_trajectories -predict writes it. replay_cofactoring,
# which runs facul_all() as las does, must find exactly those pairs with
# the strategy file. The strategies themselves depend on the timings, so
# they are not pinned.
#
# Then refine_strategy_file, on that file and on a hand-written one with
# STOP and MPQS: reordering the sides within a cell must not change the
# pairs found, and pruning must lose exactly the smooth pairs of the
# cells it drops, as it says.

FB="$1"

set -e
export LC_ALL=C

if [ "$CADO_DEBUG" ] ; then set -x ; fi

BINDIR="${CADO_NFS_BINARY_DIR?missing}/sieve"
SRCDIR="${CADO_NFS_SOURCE_DIR?missing}"
cd "${wdir:?missing}"

# three large primes on each side, so that some chains split a
# composite and have to finish it
params=(-lim0 2300000 -lim1 1200000 -lpb0 26 -lpb1 26 -mfb0 78 -mfb1 78)

# Every cell, with one cheap method per side: las spends next to nothing
# on the cofactorization, and its survivor filter keeps everything.
echo "0-78 0-78: { 0: PM1 30 100 1: PM1 30 100 }" > all.st

"$BINDIR/las" -poly "$SRCDIR/parameters/polynomials/F9.poly" -fb1 "$FB" \
    "${params[@]}" -lambda0 3.1 -lambda1 3.1 -I 12 -file-cofact all.st \
    -q0 1200000 -q1 1200030 -t 2 -dump-cofactors all.dump \
    -stats-cofact counts.txt -out las.out

# One special-q gives some 35000 pairs, most of them of 61 to 78 bits,
# which a 32-bit build factors on the slow generic arithmetic: one in
# eight is plenty here. las writes them in thread order: sort first.
LC_ALL=C sort all.dump | awk 'NR % 8 == 1' > pairs.dump

"$BINDIR/strategies/cofactoring_trajectories" -in pairs.dump \
    -out traj.txt -times times.txt "${params[@]}" -ncurves 10 -t 2

"$BINDIR/strategies/strategies_from_trajectories" -traj traj.txt \
    -times times.txt -C0 2 -out gen.st -cells gen.cells -predict predicted

"$BINDIR/strategies/replay_cofactoring" -in pairs.dump "${params[@]}" \
    -file-cofact gen.st -smooth found -t 2 > replay.out
cat replay.out

fail() { echo "$*" >&2 ; exit 1 ; }

# -smooth counts the lines of the dump, -predict those of the
# trajectories, which skip the pairs above mfb: there must be none
grep -q "skipped 0 above mfb" replay.out || fail "pairs above mfb in the dump"

ncells=$(cut -d' ' -f1,2 traj.txt | sort -u | wc -l)
# gen.cells also has the cells that the sample has no pair of
naccepted=$(awk '$5 > 0' gen.cells | wc -l)
npredicted=$(wc -l < predicted)
echo "$ncells cells, $naccepted accepted, $npredicted pairs predicted smooth"
[ "$npredicted" -gt 0 ] || fail "no pair predicted smooth"
[ "$naccepted" -gt 0 ] && [ "$naccepted" -lt "$ncells" ] || \
    fail "expected some cells accepted and some rejected"

if ! diff predicted found > /dev/null ; then
    echo "predicted and found differ (pair indices, < predicted, > found):" >&2
    diff predicted found >&2 || :
    exit 1
fi

# refine_strategy_file IN.st OUT.st [options]: refine, then replay the
# result into OUT.st.smooth; the output of refine goes to OUT.st.log
refine() {
    local in="$1" out="$2"
    shift 2
    "$BINDIR/strategies/refine_strategy_file" -in pairs.dump "${params[@]}" \
        -file-cofact "$in" -out "$out" -t 2 "$@" > "$out.log"
    cat "$out.log"
    "$BINDIR/strategies/replay_cofactoring" -in pairs.dump "${params[@]}" \
        -file-cofact "$out" -smooth "$out.smooth" -t 2 > /dev/null
}

# the pairs of $1 that $2 must have kept: all of them, or all but the
# number that the pruning reported
check_kept() {
    local before="$1" after="$2" log="$3"
    [ -z "$(comm -13 <(sort "$before") <(sort "$after"))" ] || \
        fail "$after: pairs found that $before did not find"
    local lost=0
    if [ "$log" ] ; then
        lost=$(sed -n 's/^# pruned: dropped [0-9]* cells holding \([0-9]*\) .*/\1/p' "$log")
        [ "$lost" ] || fail "$log: no pruning summary"
    fi
    local missing=$(comm -23 <(sort "$before") <(sort "$after") | wc -l)
    [ "$missing" -eq "$lost" ] || \
        fail "$after: $missing pairs lost, $lost expected"
}

refine gen.st gen-r.st -reorder
check_kept found gen-r.st.smooth
refine gen-r.st gen-rp.st -prune 0.2 -C0 0.2
check_kept found gen-rp.st.smooth gen-rp.st.log

# every cell, both sides with a head and a tail
cat > hand.st <<EOF
both = {
    0: PM1 315 2205
    1: PM1 315 2205
    0: ECM-TM12 105 3255 STOP
    1: ECM-TM12 105 3255
    0: ECM-M12 115 5775
    1: ECM-M12 115 5775 MPQS
    0: ECM-M12 125 6195
}
0-78 0-78: { both }
EOF
"$BINDIR/strategies/replay_cofactoring" -in pairs.dump "${params[@]}" \
    -file-cofact hand.st -smooth hand.smooth -t 2 > /dev/null
[ -s hand.smooth ] || fail "hand.st finds nothing"
refine hand.st hand-r.st -reorder
check_kept hand.smooth hand-r.st.smooth
grep -q "STOP" hand-r.st && grep -q "MPQS" hand-r.st || \
    fail "hand-r.st lost its STOP or MPQS"
# a threshold that drops every cell: only the pairs that need no
# factoring are left
refine hand-r.st hand-rp.st -prune 1e-6 -C0 0.1
check_kept hand.smooth hand-rp.st.smooth hand-rp.st.log

# Synthetic cofactors instead of a dump: the same trajectories whatever
# the number of threads (the times aside), and a strategy file built from
# them with the cell counts of las, which must read back and run.
for t in 1 2 ; do
    "$BINDIR/strategies/cofactoring_trajectories" -synthetic 30 -seed 1 \
        -out syn.$t.txt -times syn.times "${params[@]}" -ncurves 10 -t $t
done
cmp <(sed 's/ [0-9]\{4,\}//g' syn.1.txt) <(sed 's/ [0-9]\{4,\}//g' syn.2.txt) || \
    fail "synthetic trajectories depend on the number of threads"
"$BINDIR/strategies/strategies_from_trajectories" -traj syn.1.txt \
    -times syn.times -counts counts.txt -C0 2 -out syn.st
"$BINDIR/strategies/replay_cofactoring" -in pairs.dump "${params[@]}" \
    -file-cofact syn.st -t 2 > syn.replay
grep -q "smooth: [1-9]" syn.replay || fail "the synthetic strategy finds nothing"
