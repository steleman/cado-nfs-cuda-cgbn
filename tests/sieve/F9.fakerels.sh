#!/usr/bin/env bash

FB="$1"

set -e

if [ "$CADO_DEBUG" ] ; then set -x ; fi

BINDIR="${CADO_NFS_BINARY_DIR?missing}/sieve"
SRCDIR="${CADO_NFS_SOURCE_DIR?missing}"

WORKDIR=${wdir:?missing}

LAS="${BINDIR}/las"
FREEREL="${BINDIR}/freerel"
FAKERELS="${BINDIR}/fake_rels"

TMPSAMPLE="${WORKDIR:?missing}/cadotest.fakerel.sample"
TMPRENUMBER="${WORKDIR:?missing}/cadotest.fakerel.renumber"

cmd="${LAS} -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -lim0 100000 -lim1 100000 -lpb0 23 -lpb1 23 -mfb0 46 -mfb1 46 \
    -I 13 -fb1 ${FB} -sqside 0 \
    -q0 1000000 -q1 1001000 -random-sample 4 -dup -dup-qmin 100000,0
    -v -sync"
echo $cmd
$cmd > $TMPSAMPLE

cmd="${FREEREL} -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -renumber $TMPRENUMBER -lpb0 23 -lpb1 23 -pmax 1"
echo $cmd
$cmd

cmd="${FAKERELS} -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -lpb0 23 -lpb1 23 -q0 1000000 -q1 1001000 -sqside 0 \
    -renumber $TMPRENUMBER -sample $TMPSAMPLE"
echo $cmd
nfake=`$cmd | grep -c "^[^#]"`

# remove leading spaces (for openbsd 5.3)
nfake=`echo $nfake`
nfake_exp=185
if [ "$nfake" != "$nfake_exp" ] ; then
    echo "Wrong number of fake relations ($nfake, expected $nfake_exp)"
    exit 1
fi

# The same q range through -chunks gives the same relations.
echo "1000000 1001000 $TMPSAMPLE $WORKDIR/cadotest.fakerel.chunk" > "$WORKDIR/cadotest.fakerel.chunks"
cmd="${FAKERELS} -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -lpb0 23 -lpb1 23 -sqside 0 \
    -renumber $TMPRENUMBER -chunks $WORKDIR/cadotest.fakerel.chunks"
echo $cmd
$cmd > /dev/null
nfake=`grep -c "^[^#]" "$WORKDIR/cadotest.fakerel.chunk"`
nfake=`echo $nfake`
if [ "$nfake" != "$nfake_exp" ] ; then
    echo "Wrong number of fake relations with -chunks ($nfake, expected $nfake_exp)"
    exit 1
fi

# Two chunks with the same sample must not draw the same relations
# (they did when every chunk seeded its threads the same way: the
# second chunk then printed copies of the first one's relations, up to
# the special-q).
echo "1000000 1001000 $TMPSAMPLE $WORKDIR/cadotest.fakerel.chunkA" > "$WORKDIR/cadotest.fakerel.chunks2"
echo "1001000 1002000 $TMPSAMPLE $WORKDIR/cadotest.fakerel.chunkB" >> "$WORKDIR/cadotest.fakerel.chunks2"
cmd="${FAKERELS} -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -lpb0 23 -lpb1 23 -sqside 0 \
    -renumber $TMPRENUMBER -chunks $WORKDIR/cadotest.fakerel.chunks2"
echo $cmd
$cmd > /dev/null
for x in A B ; do
    grep "^[^#]" "$WORKDIR/cadotest.fakerel.chunk$x" | cut -d: -f1 | LC_ALL=C sort > "$WORKDIR/cadotest.fakerel.ab$x"
done
ncommon=`LC_ALL=C comm -12 "$WORKDIR/cadotest.fakerel.abA" "$WORKDIR/cadotest.fakerel.abB" | wc -l`
ncommon=`echo $ncommon`
if [ "$ncommon" != 0 ] ; then
    echo "Two chunks with the same sample share $ncommon relations"
    exit 1
fi

# Composite special-q: each relation must carry the ideals of its
# special-q (two primes in [qfac-min, qfac-max] whose product is in the
# q range). They used to get positions in the list of the ideals of that
# side instead of their indices in the table. This is checked with -dl,
# because otherwise a factor of the special-q that the relation holds
# twice cancels. -dl adds columns to the table, hence debug_renumber -dl.
cmd="${FAKERELS} -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -lpb0 23 -lpb1 23 -q0 1000000 -q1 1001000 -sqside 0 \
    -renumber $TMPRENUMBER -sample $TMPSAMPLE \
    -allow-compsq -qfac-min 400 -qfac-max 3000 -dl"
echo $cmd
$cmd > "$WORKDIR/cadotest.fakerel.compsq"
"${CADO_NFS_BINARY_DIR}/misc/debug_renumber" -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -renumber $TMPRENUMBER -dl > "$WORKDIR/cadotest.fakerel.table" 2>&1
python3 - "$WORKDIR/cadotest.fakerel.table" "$WORKDIR/cadotest.fakerel.compsq" <<'PY'
import sys, re
side, prime = {}, {}
for l in open(sys.argv[1]):
    m = re.match(r"i=0x([0-9a-f]+) .* p=0x([0-9a-f]+) .*side (\d)", l)
    if m:
        i = int(m[1], 16)
        prime[i], side[i] = int(m[2], 16), int(m[3])
n = bad = 0
for l in open(sys.argv[2]):
    if l.startswith("#"):
        continue
    ids = [int(x, 16) for x in re.split("[,:]", l.strip())[2:] if x]
    ps = sorted({prime[i] for i in ids if side[i] == 0 and 400 <= prime[i] <= 3000})
    n += 1
    if not any(1000000 <= a * b < 1001000 for k, a in enumerate(ps) for b in ps[k + 1:]):
        bad += 1
print(f"{n} relations with composite special-q, {bad} without their special-q")
sys.exit(1 if n == 0 or bad else 0)
PY

# Without the renumber table: the special-q side has one ideal per
# prime, so -virtual has the same special-q, and draws the same numbers
# of relations.
cmd="${FAKERELS} -poly ${SRCDIR}/parameters/polynomials/F9.poly \
    -lpb0 23 -lpb1 23 -q0 1000000 -q1 1001000 -sqside 0 \
    -virtual -sample $TMPSAMPLE"
echo $cmd
nfake=`$cmd | grep -c "^[^#]"`
nfake=`echo $nfake`
if [ "$nfake" != "$nfake_exp" ] ; then
    echo "Wrong number of fake relations with -virtual ($nfake, expected $nfake_exp)"
    exit 1
fi

exit 0
