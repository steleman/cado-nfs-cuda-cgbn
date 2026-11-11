#!/usr/bin/env bash
#
# The purpose of this script is to help predict the size of the matrix
# that one would obtain with a given set of parameters.
# It takes as input a polynomial file, and sieving / filtering parameters
# given as env variables.
# The CADO_NFS_SOURCE_DIR variable must point to the cado-nfs source directory
# The CADO_NFS_BINARY_DIR variable must point to the directory where cado-nfs was
# compiled.
#
# Typical usage will be something like:
#   wdir=/tmp/test A=23 lim0=... ./estimate_matsize.sh toto.poly
# The root files and the renumber file are cached, so if you change lim or
# lpb without changing wdir, you'll have to remove those cache-files
# first in wdir.
#
# If two-sided sieving is wanted, put comma-separated sides, qmin and qmax in
# the corresponding variables, for instance:
#   sqside=0,1
#   qmin=50000000,60000000
#   qmax=500000000,600000000
#
# For parameters of record size, where building the factor bases or the
# renumber table here would be out of the question, existing files can be
# given instead:
#   fb0=<file> fb1=<file>  factor bases for las (no makefb). "none" means
#                          that las gets no -fbX option for that side.
#   fbc=<file>             factor base cache for las (-fbc)
#   renumber=<file>        renumber table (no freerel)
# Other knobs: keep (purge -keep), las_sync (set it empty to run las
# without -sync), and extra_las_params, where a -hint-table or a
# -file-cofact may go.
#
# Every file that this script builds in wdir comes with a .cmd file that
# records the command that built it; a file is reused only if that
# command is unchanged. The last line of output summarizes the matrix
# size, scaled back by the shrink factor.

set -eo pipefail

## default parameters: can be overriden using env variables
## before 7151df7fe, parameters here used to correspond to a DLP-512.
## The parameters below are closer to a p80.
: ${A=23}
: ${lim0=125000}
: ${lim1=125000}
: ${maxbits=}
: ${lpb0=19}
: ${lpb1=19}
: ${mfb0=38}
: ${mfb1=38}
: ${qmin=800000}
: ${qmax=1600000}
: ${sqside=1}
: ${target_density=100}
: ${allow_compsq=true}
: ${qfac_min=200}
: ${qfac_max=100000}
: ${dlp=true}
: ${shrink_factor=1}
: ${relation_cache=}
: ${extra_las_params=""}
: ${threads=2}
: ${parallel=false}
: ${las_threads=$threads}
: ${las_parallel=$parallel}
: ${fakerels_threads=$threads}
: ${fakerels_parallel=$parallel}
: ${sampling_method=todo}       # choose either todo or random-sampling
: ${seed=0}
: ${fb0=}
: ${fb1=}
: ${fbc=}
: ${renumber=}
: ${keep=3}
: ${las_sync=-sync}

# These are normally set by cmake during tests
if ! [ -d ${CADO_NFS_BINARY_DIR:?missing} ] ; then echo "$CADO_NFS_BINARY_DIR missing" >&2 ; exit 1 ; fi
if ! [ -d ${CADO_NFS_SOURCE_DIR:?missing} ] ; then echo "$CADO_NFS_SOURCE_DIR missing" >&2 ; exit 1 ; fi


## The following can also be overriden with env variables
# the [qmin,qmax] range is split into $NCHUNKS sub-ranges
: ${NCHUNKS=3}
# for each sub-range, we call las with -random-sample $NBSAMPLE
: ${NBSAMPLE=50}

force_redo=

usage() {
    echo "Usage: $0 [options] <polyfile>"
    echo "Options: -params <param file>  read extra parameters from there (shell script)"
    echo "         -help                 show this help"
}

while [ $# -gt 0 ] ; do
if ! [[ $1 =~ ^- ]] ; then
        break
    fi
    if [ "$1" = -params ] && [ $# -gt 1 ] ; then
        if [ -f "$2" ] ; then
            . "$2"
            shift
            shift
            break
        else
            echo "$2: no such file or directory" >&2
            usage
            exit 1
        fi
    elif [ "$1" = -f ] ; then
        force_redo=1
        shift
    elif [ "$1" = -help ] ; then
        usage
        exit 0
    else
        echo "error in argument parsing at: $*" >&2
        usage
        exit 1
    fi
done

if [ $# != 1 ] ; then
    usage
    exit 1
fi

if ! [ "$CADO_NFS_BINARY_DIR" ] ; then
    echo "Error: \$CADO_NFS_BINARY_DIR must be set" >&2
    exit 1
fi

## read poly file on command line
polyfile="$1"
if [ ! -e "$1" ] ; then
    echo "Error: file $1 does not exist?" >&2
    exit 1
fi

## if wdir is not set, build one in /tmp
if [ -z ${wdir+x} ] ; then
    wdir=`mktemp -d ${TMPDIR-/tmp}/cado-nfs.est_mat.XXXXXXX`;
else
    mkdir -p $wdir || (echo "mkdir -p $wdir failed"; false) || exit 1
fi
echo "Working directory is $wdir"

test_if_has_file() {
    var="$1"
    shift
    filename="$1"
    shift
    has_file_reuse=
    has_file_command="$*"
    if ! [ -f "$filename" ] ; then
        has_file_explain=""
    elif [ "$force_redo" ] ; then
        has_file_explain="# rebuilding $filename since \$force_redo is set (either because of -f or because of outdated earlier files)"
    elif ! [ -f "$filename.cmd" ] ; then
        has_file_explain="# file $filename already in wdir, but creating command not found. not reusing file."
    elif ! diff -q "$filename.cmd" <(echo "$*") ; then
        has_file_explain="# file $filename already in wdir, but created with another command. not reusing it."
    else
        has_file_reuse=1
        has_file_command=:
        has_file_explain="# file $filename already in wdir, created with same command. reusing it."
    fi
    eval "${var}_rebuild=\$has_file_rebuild"
    eval "${var}_command=\$has_file_command"
    eval "${var}_explain=\$has_file_explain"
    eval "${var}_reuse=\$has_file_reuse"
    if [ "$has_file_reuse" ] ; then
        true
    else
        return 1
    fi
}

has_file_already() {
    testgz=1
    if [ "$1" = "-nogz" ] ; then
        testgz=
        shift
    fi
    filename="$1"
    shift

    test_if_has_file plain "$filename" "$@"

    if [ "$testgz" ] ; then
        cmdz=()
        for x in "$@" ; do
            if [ "$x" = "$filename" ] ; then
                x="$filename.gz"
            fi
            cmdz+=("$x")
        done
        test_if_has_file compressed "$filename.gz" "${cmdz[@]}"
    else
        compressed_reuse=
    fi

    reused_compressed=
    if [ "$plain_reuse" ] ; then
        echo "$plain_explain"
        return 0
    fi
    if [ "$compressed_reuse" ] ; then
        echo "$compressed_explain"
        reused_compressed=1
        return 0
    fi
    echo "$plain_explain"
    return 1
}

# Remember the command that built a file, so that it can be reused.
record_cmd() {
    filename="$1"
    shift
    echo "$*" > "$filename.cmd"
}

# Set maxbits if it is empty.
: ${maxbits:=$(((A+1)/2))}

## if wdir does not contain a rootfile, build it
rootfiles=()
for side in 0 1 ; do
    fbvar=fb$side
    limvar=lim$side
    if [ "${!fbvar}" = none ] ; then
        rootfiles+=("")
        continue
    elif [ "${!fbvar}" ] ; then
        rootfiles+=("${!fbvar}")
        continue
    fi
    rootfile="$wdir/roots$side"
    cmd=("$CADO_NFS_BINARY_DIR/sieve/makefb"
            -poly "$polyfile"
            -lim "${!limvar}"
            -maxbits "$maxbits"
            -side $side
            -t "$threads"
            -out "$rootfile")
    if ! has_file_already $rootfile "${cmd[@]}" ; then
        "${cmd[@]}"
        record_cmd "$rootfile" "${cmd[@]}"
    elif [ "$reused_compressed" ] ; then
        rootfile="$rootfile.gz"
    fi
    rootfiles+=("$rootfile")
done
fb_las=()
for side in 0 1 ; do
    if [ "${rootfiles[$side]}" ] ; then
        fb_las+=(-fb$side "${rootfiles[$side]}")
    fi
done
if [ "$fbc" ] ; then
    fb_las+=(-fbc "$fbc")
fi

dl_flag=()
if [ "$dlp" == "true" ] ; then
    dl_flag=(-dl)
fi

## if wdir does not contain a renumber table, build it
if [ "$renumber" ] ; then
    renumberfile="$renumber"
else
    renumberfile=$wdir/renumber
    cmd=("$CADO_NFS_BINARY_DIR/sieve/freerel" -poly "$polyfile" -renumber
        "$renumberfile" -pmax 1 -lpb0 "$lpb0" -lpb1 "$lpb1" "${dl_flag[@]}"
        -t "$threads")
    if ! has_file_already $renumberfile "${cmd[@]}" ; then
        "${cmd[@]}"
        record_cmd "$renumberfile" "${cmd[@]}"
    elif [ "$reused_compressed" ] ; then
        renumberfile="$renumberfile.gz"
    fi
fi

## deal with composite special-q's
compsq=""
if [ "$allow_compsq" == "true" ] ; then
    compsq_fake="-allow-compsq -qfac-min ${qfac_min} -qfac-max ${qfac_max}"
    compsq_las="-allow-largesq ${compsq_fake}"
fi

## How many sides to sieve?
# convert sqside, qmin and qmax to arrays
IFS=',' read -ra array <<< "$sqside"
sqside=(${array[@]})
IFS=',' read -ra array <<< "$qmax"
qmax=(${array[@]})
IFS=',' read -ra array <<< "$qmin"
qmin=(${array[@]})

nsides=${#sqside[@]}
echo "We sieve on $nsides sides."
if [ "${#sqside[@]}" != "${#qmax[@]}" -o "${#sqside[@]}" != "${#qmin[@]}" ] ; then
    echo "For multi-side sieving, length of sqside, qmin and qmax arrays must agree"
    exit 1;
fi
for i in `seq 0 $((nsides-1))`; do
    side=${sqside[$i]}
    qm=${qmax[$i]}
    lpb=lpb$side
    qqmax=`echo "2 ^ ${!lpb}" | bc`
    if [ "$allow_compsq" == "false" -a $qqmax -le $qm ] ; then
        echo "Error on side $side: qmax should be less then lpb"
        exit 1
    fi
    echo "  Side $side: qmin=${qmin[$i]} qmax=$qm"
done

## Background jobs: wait for the oldest one, and stop if it failed.
processes=()
wait_first() {
    if ! wait "${processes[0]}" ; then
        echo "Error: a background job failed, see above" >&2
        exit 1
    fi
    processes=("${processes[@]:1}")
}

## Sampling / faking on each side
fakefiles=()
if [ $nsides == 1 ] ; then
    if [ ${sqside[0]} == 0 ] ; then
        dupqmin="${qmin[0]},0"
    else
        dupqmin="0,${qmin[0]}"
    fi
elif [ $nsides == 2 ] ; then
    dupqmin="${qmin[0]},${qmin[1]}"
else
    echo "Can't deal with more than 2 sides yet."
    exit 1
fi
for i in `seq 0 $((nsides-1))`; do
    side=${sqside[$i]}
    echo "******************************************************"
    echo "Dealing with side $side (creating samples with las)..."
    side_qmax=${qmax[$i]}
    side_qmin=${qmin[$i]}
    qrange=$(((side_qmax-side_qmin)/NCHUNKS))

    ## sample with real sieving and build fake rels
    processes=()
    for i in `seq 1 $NCHUNKS`; do
        (
            q0=$((side_qmin + (i-1)*qrange))
            q1=$((side_qmin + i*qrange))
            echo "Sampling in qrange=[$q0,$q1] ; sampling method: $sampling_method"
            samplebase=$wdir/sample.side${side}.${q0}-${q1}

            cmd0=($CADO_NFS_BINARY_DIR/sieve/las -A $A -poly $polyfile -q0 $q0 -q1 $q1 
              -lim0 $lim0 -lim1 $lim1 -lpb0 $lpb0 -lpb1 $lpb1 -sqside $side 
              -mfb0 $mfb0 -mfb1 $mfb1 $compsq_las
              "${fb_las[@]}"
              -t $las_threads $las_sync -v -dup -dup-qmin $dupqmin
                          $extra_las_params)

            case "$sampling_method" in
                random-sample)
                    cmd=("${cmd0[@]}" -random-sample $NBSAMPLE -seed $seed)
                    ;;
                todo)
                    generate=("${cmd0[@]}" -print-todo-list)
                    completelist=$samplebase.completelist
                    todolist=$samplebase.todolist
                    if ! has_file_already -nogz "$completelist" "${generate[@]}" ; then
                        echo "Preparing complete list of special-q in $completelist"
                        echo "${generate[@]}"
                        "${generate[@]}" | grep '^[0-9]' > "$completelist"
                        record_cmd "$completelist" "${generate[@]}"
                    fi
                    random_gen=($CADO_NFS_SOURCE_DIR/tests/linalg/bwc/perlrandom.pl 0 $seed)
                    # busybox (at least on alpine linux) provides shuf
                    # and sort,
                    if type -p shuf > /dev/null 2>&1 && shuf --version 2>&1 | grep -q GNU.coreutils; then
                        fshuf() { shuf --random-source=<("${random_gen[@]}") "$@" ; }
                    elif type -p shuf > /dev/null 2>&1 ; then
                        fshuf() { shuf "$@" ; }
                    elif ! sort --version 2>&1 | grep -q -e -R ; then
                        fshuf() { sort -R "$@" ; }
                    else
                        # really a last resort.
                        fshuf() { cat "$@" | rev | sort | rev ; }
                    fi
                    # not a pipe into head: with pipefail, the SIGPIPE
                    # that the shuffle gets when head is done would count
                    # as a failure
                    fshuf $completelist > $todolist.shuffled
                    head -n $NBSAMPLE $todolist.shuffled > $todolist
                    rm -f $todolist.shuffled
                    cmd=("${cmd0[@]}" -todo $todolist )
                    ;;
                *)
                    echo "unsupported sampling method" >&2
                    exit 1
                    ;;
            esac
            if [ "${relation_cache}" ] ; then
                cmd+=(-relation-cache "$relation_cache")
            fi
            echo "${cmd[@]}"
            file=$samplebase
            if ! has_file_already $file "${cmd[@]}" ; then
                "${cmd[@]}" > $file
                record_cmd "$file" "${cmd[@]}"
            fi
        ) &
        processes+=($!)
        bound=$las_parallel
        if [ $las_parallel = false ] ; then bound=1 ; fi
        if [ $las_parallel != true ] && [ ${#processes[@]} -ge $bound ] ; then
            wait_first
        fi
    done
    while [ ${#processes[@]} -gt 0 ] ; do wait_first ; done
done

for i in `seq 0 $((nsides-1))`; do
    side=${sqside[$i]}
    side_qmax=${qmax[$i]}
    side_qmin=${qmin[$i]}
    qrange=$(((side_qmax-side_qmin)/NCHUNKS))
    echo "****************************************************"
    echo "Dealing with side $side (creating fake relations)..."
    for i in `seq 1 $NCHUNKS`; do
        (
            q0=$((side_qmin + (i-1)*qrange))
            q1=$((side_qmin + i*qrange))
            samplebase=$wdir/sample.side${side}.${q0}-${q1}
            echo "  Building fake relations in qrange=[$q0,$q1]"
            cmd=($CADO_NFS_BINARY_DIR/sieve/fake_rels -poly $polyfile -lpb0 $lpb0 -lpb1 $lpb1
              -q0 $q0 -q1 $q1 -sqside $side $compsq_fake
              -sample $samplebase
              -shrink-factor $shrink_factor
              "${dl_flag[@]}"
              -t $fakerels_threads
              -renumber $renumberfile)
            echo "${cmd[@]}"
            file=$wdir/fakerels.side${side}.${q0}-${q1}
            if ! has_file_already $file "${cmd[@]}" ; then
                "${cmd[@]}" > $file
                record_cmd "$file" "${cmd[@]}"
            fi
        ) &
        processes+=($!)
        if [ $fakerels_parallel == "false" ] ; then
            wait_first
        fi
    done
    while [ ${#processes[@]} -gt 0 ] ; do wait_first ; done
    for i in `seq 1 $NCHUNKS`; do
        q0=$((side_qmin + (i-1)*qrange))
        q1=$((side_qmin + i*qrange))
        fakefiles+=("$wdir/fakerels.side${side}.${q0}-${q1}")
    done
done
nrels=`cat ${fakefiles[@]} | grep -v "^#" | wc -l`
echo "We have $nrels fake relations"

## Filtering
# purge
cmd=($CADO_NFS_BINARY_DIR/filter/purge -out $wdir/purged.gz -keep $keep
    -t $threads
    ${fakefiles[@]})
file=$wdir/purged.gz
if ! has_file_already $file "${cmd[@]}" ; then
    "${cmd[@]}" 2>&1 | tee $wdir/purge.log
    record_cmd "$file" "${cmd[@]}"
fi

# Did we get a positive excess?
if (grep "number of rows < number of columns + keep" $wdir/purge.log > /dev/null); then
    echo "Negative excess: no way to build a matrix!"
    echo "You'll have to change the parameters"
    exit 1
fi

# merge
if [ "$dlp" == "true" ] ; then
    cmd=($CADO_NFS_BINARY_DIR/filter/merge-dl -mat $wdir/purged.gz -out $wdir/history.gz 
        -skip 0 -target_density $target_density -t $threads)
    file=$wdir/history.gz
    if ! has_file_already $file "${cmd[@]}" ; then
        "${cmd[@]}" 2>&1 | tee $wdir/merge-dl.log
        record_cmd "$file" "${cmd[@]}"
    fi
    mergelog=$wdir/merge-dl.log
else
    cmd=($CADO_NFS_BINARY_DIR/filter/merge -mat $wdir/purged.gz -out $wdir/history.gz
        -skip 32 -target_density $target_density -t $threads)
    file=$wdir/history.gz
    if ! has_file_already $file "${cmd[@]}" ; then
        "${cmd[@]}" 2>&1 | tee $wdir/merge.log
        record_cmd "$file" "${cmd[@]}"
    fi
    mergelog=$wdir/merge.log
fi

## Summary
purged=$(grep -o 'nrows=[0-9]* ncols=[0-9]*' $wdir/purge.log | tail -n 1)
final=$(grep '^Final matrix has' $mergelog | tail -n 1)
N=$(echo "$final" | sed -n 's/.*N=\([0-9]*\).*/\1/p')
W=$(echo "$final" | sed -n 's/.* W=\([0-9]*\).*/\1/p')
if [ "$N" ] && [ "$W" ] ; then
    echo "# estimate_matsize: $nrels fake relations; after purge: $purged; after merge: N=$N W=$W W/N=$((W/N)); shrink factor $shrink_factor, i.e. N=$((N*${shrink_factor%.*})) at full scale"
else
    echo "Error: could not find the final matrix size in $mergelog" >&2
    exit 1
fi
