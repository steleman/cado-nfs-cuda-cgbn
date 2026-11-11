#!/usr/bin/env bash

# A hybrid schedule: prime special-q from lim1 up to X, composite ones
# (factors in [qfac-min, qfac-max]) from X on. The duplicate check of a
# relation found by a composite special-q must consider both kinds: on
# the relations of a composite run, the duplicates for the hybrid
# schedule are those for the prime part plus those for the composite
# part. On the relations of a prime run, the composite part (above X)
# changes nothing.

if [ "$CADO_DEBUG" ] ; then set -x ; fi
set -e

: ${wdir:?missing}
: ${LAS_BINARY:?missing}
: ${DUPSUP_BINARY:?missing}
: ${poly:?missing}
: ${fb1:?missing}

X=200000000
common=(-poly $poly -fb1 $fb1 -lim0 2300000 -lim1 1200000 -I 12
    -lpb0 28 -lpb1 28 -mfb0 56 -mfb1 56 -lambda0 2.1 -lambda1 2.1
    -ncurves0 10 -ncurves1 10 -adjust-strategy 0)
compsq=(-qfac-min 5000 -qfac-max 100000)

"$LAS_BINARY" "${common[@]}" -sqside 1 -q0 250000000 -q1 250500000 \
    -allow-compsq "${compsq[@]}" -random-sample 3 -seed 1 \
    -v -out $wdir/rels.composite
"$LAS_BINARY" "${common[@]}" -sqside 1 -q0 150000000 -nq 1 \
    -v -out $wdir/rels.prime

dupes() { # FILE OUTPUT dupsup-arguments...
    local in=$1 out=$2
    shift 2
    # dupsup reads the special-q from the relations, but wants a -q0
    "$DUPSUP_BINARY" "${common[@]}" -q0 $X -q1 $X -dup 1 "$@" -out $wdir/$out $in
    grep '^# DUPE ' $wdir/$out | cut -d: -f1 | sed 's/^# DUPE //' | LC_ALL=C sort > $wdir/$out.dupe
}

hybrid=(-allow-compsq 1 "${compsq[@]}" -dup-qmin 0,1200000 -dup-qmax-prime 0,$X -dup-qmin-composite 0,$X)

dupes $wdir/rels.composite c.hybrid "${hybrid[@]}"
dupes $wdir/rels.composite c.prime -dup-qmin 0,1200000 -dup-qmax 0,$X
dupes $wdir/rels.composite c.composite -allow-compsq 1 "${compsq[@]}" -dup-qmin 0,$X
LC_ALL=C sort -u $wdir/c.prime.dupe $wdir/c.composite.dupe > $wdir/c.union.dupe
n_hybrid=$(wc -l < $wdir/c.hybrid.dupe)
n_prime=$(wc -l < $wdir/c.prime.dupe)
n_composite=$(wc -l < $wdir/c.composite.dupe)
echo "composite run: $n_hybrid duplicates for the hybrid schedule, $n_prime for its prime part, $n_composite for its composite part"
if ! cmp -s $wdir/c.hybrid.dupe $wdir/c.union.dupe ; then
    echo "the duplicates for the hybrid schedule are not those of its two parts" >&2
    exit 1
fi
if ! LC_ALL=C comm -23 $wdir/c.prime.dupe $wdir/c.composite.dupe | grep -q . ; then
    echo "the prime part finds nothing that the composite part does not: not a test" >&2
    exit 1
fi

dupes $wdir/rels.prime p.hybrid "${hybrid[@]}"
dupes $wdir/rels.prime p.prime -dup-qmin 0,1200000 -dup-qmax 0,$X
if ! cmp -s $wdir/p.hybrid.dupe $wdir/p.prime.dupe ; then
    echo "on a prime run, the composite part of the schedule changed the duplicates" >&2
    exit 1
fi
echo "prime run: $(wc -l < $wdir/p.hybrid.dupe) duplicates either way"
