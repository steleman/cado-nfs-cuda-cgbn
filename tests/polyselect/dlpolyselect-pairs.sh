#!/usr/bin/env bash

# Every pair printed by dlpolyselect must have a common root modulo N:
# score reads each of them through cado_poly, which checks this. With
# -bound 3, several f have two or more roots modulo N, so that the g of
# a root other than the first one are checked too. (With a much smaller
# N, some g are not squarefree, and alpha recurses forever on them.)

if [ "$CADO_DEBUG" ] ; then set -x ; fi
set -e

: ${wdir:?missing}
: ${DLPOLYSELECT:?missing}
: ${SCORE:?missing}

"$DLPOLYSELECT" "$@" > $wdir/out
awk -v d=$wdir '/^n:/ { n++ } n && !/^#/ && NF { print > (d "/pair." n ".poly") }' $wdir/out
npairs=$(ls $wdir/pair.*.poly | wc -l)
if [ "$npairs" -lt 2 ] ; then
    echo "only $npairs pair(s) printed" >&2
    exit 1
fi
for f in $wdir/pair.*.poly ; do
    if ! "$SCORE" $f > /dev/null 2> $wdir/score.err ; then
        echo "no common root modulo N in $f:" >&2
        cat $f $wdir/score.err >&2
        exit 1
    fi
done
echo "$npairs pairs checked"
