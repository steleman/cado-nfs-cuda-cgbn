#!/usr/bin/env bash

# build_cofactoring_strategies splits its input file into one family per
# method name that facul knows, and indexes the first element of each
# before handing them to the convex hull. A file that mentions no method
# of some family must therefore be diagnosed, not indexed into: check
# that each of them is.
#
# The input is a pinned sample file rather than something produced by the
# survey/bench chain: bench_cofactoring_methods selects methods from
# wall-clock measurements taken on the spot, so that chain does not give
# the same output twice.

set -e

: ${CADO_NFS_SOURCE_DIR:?missing}
: ${CADO_NFS_BINARY_DIR:?missing}
: ${wdir:?missing}

BUILD="$CADO_NFS_BINARY_DIR/sieve/strategies/build_cofactoring_strategies"
PATTERNS="$CADO_NFS_BINARY_DIR/sieve/strategies/list_factorization_patterns"
sample="$CADO_NFS_SOURCE_DIR/tests/sieve/strategies/sample_methods.txt"

lim0=1024
lpb0=16
r0=22

cd "$wdir"

$PATTERNS -lim $lim0 -mfb $r0 -out decomp_${lim0}_${r0}

run_oneside() {
    # the -oneside mode of build_cofactoring_strategies, on the method file
    # given as $1
    mkdir -p out
    $BUILD -oneside -lim0 $lim0 -lpb0 $lpb0 -r0 $r0 -ncurves 3 \
        -in "$1" -decomp decomp_${lim0}_${r0} -out out
}

# a complete file must still go through, and produce the output file.
run_oneside "$sample"
if ! [ -f out/strategies${lim0}_${r0} ] ; then
    echo "build_cofactoring_strategies produced no strategies file on a complete input" >&2
    exit 1
fi

# Each case below is an extended regexp matching the lines of one family,
# and the name the binary is expected to use for it -- the same name, now
# that the files spell the method out. Removing PP1-27 or PP1-65 alone is
# fine, so the two go together: only their union is used.
cases=(
    "PM1 :PM1"
    "PP1-27 |PP1-65 :PP1-27 or PP1-65"
    "ECM-B12 :ECM-B12"
    "ECM-M12 :ECM-M12"
    "ECM-M16 :ECM-M16"
    "ECM-TM12 :ECM-TM12"
)

rc=0
for c in "${cases[@]}" ; do
    family="${c%%:*}"
    name="${c#*:}"
    grep -E -v "^($family)" "$sample" > amputated.txt
    rm -rf out
    status=0
    run_oneside amputated.txt > stdout.txt 2> stderr.txt || status=$?
    if [ $status = 0 ] ; then
        echo "an input file with no $name method was accepted" >&2
        rc=1
    elif [ $status -ge 128 ] ; then
        # not merely cosmetic: a runtime that kills the process on an
        # uncaught exception need not print what() first, and libc++
        # does not.
        echo "died on a signal on a file with no $name method" >&2
        rc=1
    elif ! grep -q "contains no $name method" stderr.txt ; then
        echo "failed on a file with no $name method, but without saying so:" >&2
        tail -3 stderr.txt >&2
        rc=1
    fi
done

exit $rc
