#!/usr/bin/env bash

# Pin what can be pinned of survey_cofactoring_methods and bench_cofactoring_methods.
#
# These two measure wall-clock time and write it into their output, so
# they never produce the same bytes twice. With -seed fixed, though,
# everything *else* they write is reproducible: the methods they select
# and the probabilities they measure. This test therefore compares their
# output with the timing column stripped off.
#
# Not every mode survives that treatment:
#
#   survey_cofactoring_methods (no -ch), bench_cofactoring_methods -p -t     reproducible once the timings are
#                                   stripped -- pinned here.
#   bench_cofactoring_methods -p, bench_cofactoring_methods -f, survey_cofactoring_methods     no measurement involved, or only a
#   -fch                            pinned file as input: reproducible as
#                                   they stand -- pinned verbatim here.
#   survey_cofactoring_methods -ch                         NOT reproducible, even stripped: the
#                                   convex hull ranks methods by
#                                   (probability, time), so timing noise
#                                   changes which methods survive. Eight
#                                   runs at one seed gave three different
#                                   answers, with hulls of different
#                                   sizes. Only smoke-checked below; the
#                                   same hull code is pinned properly
#                                   through -fch, which reads a fixed file.
#
# Re-run with REGEN=1 to rewrite the references after a deliberate change.

set -e

# Everything below compares text, which the locale can change under our
# feet.
export LC_ALL=C

: ${CADO_NFS_SOURCE_DIR:?missing}
: ${CADO_NFS_BINARY_DIR:?missing}
: ${wdir:?missing}

SURVEY="$CADO_NFS_BINARY_DIR/sieve/strategies/survey_cofactoring_methods"
BENCH="$CADO_NFS_BINARY_DIR/sieve/strategies/bench_cofactoring_methods"
here="$CADO_NFS_SOURCE_DIR/tests/sieve/strategies"
ref="$here/expected"
sample="$here/sample_methods.txt"

# A small sieve region, few sizes and few tests: this must stay short in
# a coverage build too, which is about 25 times slower.
region="-b1min 100 -b1max 200 -b1step 100 -cmin 10 -cmax 20 -cstep 10"

cd "$wdir"
rc=0

# drop the third |-delimited field, which holds the measured timings
strip_timings() { awk -F'|' '{print $1 "|" $2 "|"}' "$1" ; }

compare() { # <name> <actual> <how it was produced>
    local name="$1" actual="$2" how="$3"
    # The reference carries a header saying how it was made, written from
    # the same place that runs the command, so the two cannot drift apart.
    # It is a comment in the format being pinned, so the reference stays
    # valid input for the reader that produced it.
    { echo "# $how"
      echo "# regenerate: REGEN=1 $(basename "$0")"
      cat "$actual"
    } > "annotated_$name"

    if [ "$REGEN" ] ; then
        mkdir -p "$ref"
        cp "annotated_$name" "$ref/$name"
        echo "REGEN: wrote $ref/$name"
        return
    fi
    if [ ! -f "$ref/$name" ] ; then
        echo "missing reference $ref/$name (re-run with REGEN=1)" >&2
        rc=1
        return
    fi
    if ! diff -u "$ref/$name" "annotated_$name" > "diff_$name.txt" ; then
        echo "$name: output differs from the pinned reference" >&2
        head -40 "diff_$name.txt" >&2
        rc=1
    fi
}

############################################################
echo "### survey_cofactoring_methods, one family at a time, timings stripped"
# every family survey_cofactoring_methods knows; PP1-65 and ECM-TM12 used
# not to be here, which left two of them unpinned
for m in PM1 PP1-27 PP1-65 ECM-B12 ECM-M12 ECM-M16 ECM-TM12 ; do
    $SURVEY -lb 16 -ub 17 -m $m $region -seed 1 -out survey_$m
    strip_timings survey_$m > survey_$m.notime
    compare survey_$m.notime survey_$m.notime \
        "survey_cofactoring_methods -lb 16 -ub 17 -m $m $region -seed 1, timings stripped"
done

############################################################
echo "### survey_cofactoring_methods -fch (convex hull of a pinned file)"
$SURVEY -fch -fch_in "$sample" -fch_out survey_fch.txt > /dev/null
compare survey_fch.txt survey_fch.txt \
    "survey_cofactoring_methods -fch -fch_in sample_methods.txt"

############################################################
echo "### survey_cofactoring_methods -ch (smoke only, see the header)"
$SURVEY -lb 16 -ub 17 -m ECM-M12 $region -seed 1 -ch -out survey_ch.txt
if [ ! -s survey_ch.txt ] ; then
    echo "survey_cofactoring_methods -ch produced an empty hull" >&2
    rc=1
fi
# whatever survives must be one of the methods that were swept
$SURVEY -lb 16 -ub 17 -m ECM-M12 $region -seed 1 -out survey_noch.txt
while read -r name b1 b2 rest ; do
    if ! cut -d'|' -f1 survey_noch.txt | grep -qx "$name $b1 $b2 " ; then
        echo "survey_cofactoring_methods -ch returned a method that was never swept: $name $b1 $b2" >&2
        rc=1
    fi
done < <(cut -d'|' -f1 survey_ch.txt)

############################################################
echo "### bench_cofactoring_methods -p (probabilities only: no measurement in the output)"
$BENCH -in "$sample" -p -lb 10 -N 30 -seed 2 -out bench_p.txt
compare bench_p.txt bench_p.txt \
    "bench_cofactoring_methods -in sample_methods.txt -p -lb 10 -N 30 -seed 2"

############################################################
echo "### bench_cofactoring_methods -p -t, timings stripped"
$BENCH -in "$sample" -p -t -lb 10 -N 30 -seed 2 -out bench_pt
strip_timings bench_pt > bench_pt.notime
compare bench_pt.notime bench_pt.notime \
    "bench_cofactoring_methods -in sample_methods.txt -p -t -lb 10 -N 30 -seed 2, timings stripped"

# -p alone and -p -t must agree on the probabilities they measure: same
# seed, same input, and the timing pass does not touch them.
if ! diff -q <(strip_timings bench_p.txt) bench_pt.notime > /dev/null ; then
    echo "bench_cofactoring_methods -p and bench_cofactoring_methods -p -t disagree on the probabilities" >&2
    diff <(strip_timings bench_p.txt) bench_pt.notime | head -10 >&2
    rc=1
fi

############################################################
echo "### bench_cofactoring_methods -f (filtering a pinned file)"
$BENCH -in "$sample" -f 3 -lb 10 -out bench_f3.txt
compare bench_f3.txt bench_f3.txt \
    "bench_cofactoring_methods -in sample_methods.txt -f 3 -lb 10"

exit $rc
