#!/usr/bin/env bash

# See https://sympa.inria.fr/sympa/arc/cado-nfs/2020-09/msg00020.html

: ${CADO_NFS_BINARY_DIR:?missing}
: ${CADO_NFS_SOURCE_DIR:?missing}

FREEREL=${CADO_NFS_BINARY_DIR}/sieve/freerel
FAKERELS=${CADO_NFS_BINARY_DIR}/sieve/fake_rels

TMPRENUMBER="${wdir:?missing}/cadotest.fakerel.renumber"
TMPRELS="${wdir:?missing}/cadotest.fakerel.rels"

cmd="${FREEREL} -poly ${CADO_NFS_SOURCE_DIR}/parameters/polynomials/c60.poly \
    -renumber $TMPRENUMBER -lpb0 18 -lpb1 19 -pmax 1"
echo $cmd
$cmd

# The following was segfaulting
cmd="${FAKERELS} -poly ${CADO_NFS_SOURCE_DIR}/parameters/polynomials/c60.poly \
    -lpb0 18 -lpb1 19 -q0 61961 -q1 62961 -sqside 1 -shrink-factor 1 \
    -renumber $TMPRENUMBER -sample ${CADO_NFS_SOURCE_DIR}/tests/sieve/c60.sample"
echo $cmd
$cmd

# Relations that go to a GF(2) matrix may not have the same ideal twice.
# A shrink factor makes this a demanding check, since it maps ideals that
# sit on different sides onto a common index.
cmd="${FAKERELS} -poly ${CADO_NFS_SOURCE_DIR}/parameters/polynomials/c60.poly \
    -lpb0 18 -lpb1 19 -q0 61961 -q1 62961 -sqside 1 -shrink-factor 10 \
    -renumber $TMPRENUMBER -sample ${CADO_NFS_SOURCE_DIR}/tests/sieve/c60.sample"
echo $cmd
$cmd > $TMPRELS

nrels=$(grep -c "^[^#]" $TMPRELS)
echo "$nrels relations to check"
if [ "$nrels" -lt 100 ] ; then
    echo "Too few relations for the check below to be meaningful"
    exit 1
fi

read -s -r -d '' perl_code <<- 'EOF'
	next if /^#/;
	chomp;
	my (undef, $ideals) = split /:/, $_, 2;
	next unless defined $ideals;
	my %seen;
	for my $h (split /,/, $ideals) {
	    die "ideal $h appears twice in relation: $_\n" if $seen{$h}++;
	}
EOF
if ! perl -ne "$perl_code" $TMPRELS ; then
    echo "fake_rels emitted a relation with a repeated ideal"
    exit 1
fi
