#!/usr/bin/env bash

if [ "$CADO_DEBUG" ] ; then
    set -x
fi
set -e

WORKDIR=${wdir:?missing}

RELSD="${WORKDIR:?missing}/rels_dupsup"
RELSR="${WORKDIR:?missing}/rels_raw"
RELSF="${WORKDIR:?missing}/rels_filtered"

PHONY_CHECK=1 RELS=$RELSD "`dirname $0`/sievetest.sh" -dup -dup-qmin 0,$lim1 "$@"
ndup_ref=`grep -wc DUPE $RELSD || :`
# remove leading spaces (for openbsd 5.3)
ndup_ref=`echo $ndup_ref`
if [ "$ndup_ref" != "$DUPE_COUNT" ] ; then
    echo "Wrong number of duplicates in las -dup"
    echo "Expected $DUPE_COUNT, got $ndup_ref"
    exit 1
fi


PHONY_CHECK=1 RELS=$RELSR "`dirname $0`/sievetest.sh" "$@"

if [ "$INTERLEAVE" ] ; then
    # What las writes with several subjobs: the special-q that run at the
    # same time start together, and their relations come after all of
    # their "# Sieving" banners. dupsup must still find the special-q of
    # each relation.
    {
        grep '^# Now sieving' "$RELSR" || :
        grep '^# Sieving' "$RELSR" || :
        grep -v '^#' "$RELSR" || :
        grep '^# Time for' "$RELSR" || :
    } > "$RELSR.interleaved"
    RELSR="$RELSR.interleaved"
fi

# with SUBLAT set, las sieved with -sublat $SUBLAT, which dupsup must know
PHONY_CHECK=1 LAS_BINARY=$DUPSUP_BINARY RELS=$RELSF "`dirname $0`/sievetest.sh" -skew 1.0 ${SUBLAT:+-sublat $SUBLAT} "${RELSR}"
ndup=`grep -wc DUPE $RELSF || :`
# remove leading spaces (for openbsd 5.3)
ndup=`echo $ndup`
if [ "$ndup" != "$DUPE_COUNT" ] ; then
    echo "Wrong number of duplicates in dupsup"
    echo "Expected $DUPE_COUNT, got $ndup"
    exit 1
fi
