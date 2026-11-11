#!/usr/bin/env bash

# Pin the output of the deterministic half of the strategies chain.
#
# build_cofactoring_strategies, in all four of its modes, and choose_cofactoring_strategies draw no random numbers: they
# are pure functions of their input files. This test runs them on fixed
# inputs and compares the result against references committed beside it,
# so that reworking sieve/strategies has a net that catches any change of
# behaviour rather than only a change of exit status.
#
# The other half of the chain -- survey_cofactoring_methods and bench_cofactoring_methods -- measures wall-clock
# time and cannot be pinned this way; test_strategies_pinned_notime.sh
# covers it modulo the timing columns.
#
# After a deliberate change of behaviour, re-run with REGEN=1 to rewrite
# the references in the source tree, and read the diff before committing.

set -e

# Everything below compares text and sorts file names, both of which the
# locale can change under our feet.
export LC_ALL=C

: ${CADO_NFS_SOURCE_DIR:?missing}
: ${CADO_NFS_BINARY_DIR:?missing}
: ${wdir:?missing}

BUILD_ST="$CADO_NFS_BINARY_DIR/sieve/strategies/build_cofactoring_strategies"
PATTERNS="$CADO_NFS_BINARY_DIR/sieve/strategies/list_factorization_patterns"
CHOOSE_ST="$CADO_NFS_BINARY_DIR/sieve/strategies/choose_cofactoring_strategies"
here="$CADO_NFS_SOURCE_DIR/tests/sieve/strategies"
ref="$here/expected"
sample="$here/sample_methods.txt"

lim=1024
lpb=16
mfb=26
ncurves=3

# fbb = ceil(log2(lim+1)) = 11 here, so build_cofactoring_strategies treats a cofactor of fewer
# than 2*fbb-1 = 21 bits as prime and hands it a trivial strategy. The
# band [21, mfb] is where the generator actually does work, and it is
# where the references below are taken.
rmin=21

# The pairs pinned verbatim: the two ends of the diagonal, one cell off
# it, and one strongly asymmetric cell.
pinned_pairs="21_21 21_26 24_25 26_26"

cd "$wdir"
rc=0

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

label() { # concatenate files, each preceded by its basename
    # The label is a comment in the format being concatenated, so that
    # each reference below is still valid input for the reader that
    # produced it, and not merely something to diff against.
    local f
    for f in "$@" ; do
        echo "# $(basename "$f")"
        cat "$f"
    done
}

############################################################
echo "### list_factorization_patterns"
mkdir -p dec
for r in $(seq $rmin $mfb) ; do
    $PATTERNS -lim $lim -mfb $r -out dec/decomp_${lim}_$r
done
label dec/decomp_${lim}_* > decomps.txt
compare decomps.txt decomps.txt \
    "list_factorization_patterns -lim $lim -mfb <r>, r in [$rmin, $mfb]"

############################################################
echo "### build_cofactoring_strategies -oneside"
mkdir -p pre
for r in $(seq $rmin $mfb) ; do
    $BUILD_ST -oneside -lim0 $lim -lpb0 $lpb -r0 $r -ncurves $ncurves \
        -in "$sample" -decomp dec/decomp_${lim}_$r -out pre
done
# the trivial sizes below rmin are needed by the -merge step further down
for r in $(seq 0 $((rmin-1))) ; do
    $BUILD_ST -oneside -lim0 $lim -lpb0 $lpb -r0 $r -ncurves $ncurves \
        -in "$sample" -decomp /dev/null -out pre
done
label $(for r in $(seq $rmin $mfb) ; do echo pre/strategies${lim}_$r ; done) > oneside.txt
compare oneside.txt oneside.txt \
    "build_cofactoring_strategies -oneside -lim0 $lim -lpb0 $lpb -r0 <r> -ncurves $ncurves -in sample_methods.txt, r in [$rmin, $mfb]"

############################################################
echo "### build_cofactoring_strategies -merge (one pair at a time)"
mkdir -p pair
for r0 in $(seq $rmin $mfb) ; do
    for r1 in $(seq $rmin $mfb) ; do
        $BUILD_ST -merge -lim0 $lim -lim1 $lim -r0 $r0 -r1 $r1 -in pre -out pair
    done
done
label $(for p in $pinned_pairs ; do echo pair/strategies_$p ; done) > pairs.txt
compare pairs.txt pairs.txt \
    "build_cofactoring_strategies -merge -lim0 $lim -lim1 $lim -r0 <a> -r1 <b>, for $pinned_pairs"

############################################################
echo "### build_cofactoring_strategies, default mode (whole matrix in one go)"
mkdir -p mat
$BUILD_ST -lim0 $lim -lpb0 $lpb -mfb0 $mfb -lim1 $lim -lpb1 $lpb -mfb1 $mfb \
    -ncurves $ncurves -in "$sample" -decomp dec -out mat > /dev/null

# -ne and not !=, because wc pads its output on the BSDs and the string
# comparison would then never match
n=$(ls mat | wc -l)
if [ "$n" -ne $(( (mfb+1) * (mfb+1) )) ] ; then
    echo "build_cofactoring_strategies default mode wrote $n files, expected $(( (mfb+1)*(mfb+1) ))" >&2
    rc=1
fi
label $(for p in $pinned_pairs ; do echo mat/strategies_$p ; done) > matrix.txt
compare matrix.txt matrix.txt \
    "build_cofactoring_strategies (no mode switch), same cells as pairs.txt"

# the whole matrix, as one digest: too large to pin verbatim, but a
# change anywhere in it should still be noticed. The file list is built
# in numeric order rather than by sorting names, so that the digest does
# not depend on how the locale collates them.
for r0 in $(seq 0 $mfb) ; do
    for r1 in $(seq 0 $mfb) ; do
        echo "strategies_${r0}_${r1}"
    done
done > matrix.list
( cd mat && cat $(cat ../matrix.list) ) | sha256sum | cut -d' ' -f1 \
    > matrix.sha256
compare matrix.sha256 matrix.sha256 \
    "sha256 over all $(( (mfb+1) * (mfb+1) )) cells of that matrix"

############################################################
echo "### the two routes to a pair must agree on the method chains"
# They are not byte-identical: -merge reads the one-sided strategies back
# from a file that stores them with six decimals, so the probabilities
# and times it derives differ from the in-memory ones in the last digit.
# The chains of methods themselves must match exactly.
for r0 in $(seq $rmin $mfb) ; do
    for r1 in $(seq $rmin $mfb) ; do
        a=pair/strategies_${r0}_${r1}
        b=mat/strategies_${r0}_${r1}
        if ! diff -q <(grep -v '^Probability:\|^Time:' $a) \
                     <(grep -v '^Probability:\|^Time:' $b) > /dev/null ; then
            echo "cell ($r0,$r1): -merge and the default mode disagree on the methods" >&2
            rc=1
        fi
    done
done

############################################################
echo "### choose_cofactoring_strategies"
# A real cofactor distribution is sparse -- most size pairs never occur --
# so the reference stays small without losing any of the selection logic.
for r0 in 21 23 24 26 ; do
    for r1 in 21 23 24 26 ; do
        echo "$r0 $r1 $(( 1 + (r0 * 7 + r1 * 3) % 11 )) 1"
    done
done > cofactors.stats

# two values of -t, to land on two different points of the slope
# selection in compute_best_strategy().
for t in 0.01 100 ; do
    $CHOOSE_ST -st mat -dist cofactors.stats -t $t -mfb0 $mfb -mfb1 $mfb \
        -out final_st_$t > /dev/null
    compare final_st_$t final_st_$t \
        "choose_cofactoring_strategies -st <that matrix> -dist <fixed> -t $t -mfb0 $mfb -mfb1 $mfb"
done

exit $rc
