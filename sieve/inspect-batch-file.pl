#!/usr/bin/env perl

use strict;
use warnings;

use Fcntl;
use Fcntl 'SEEK_SET';

# This script lists the contents of an .batch file (product trees of prime
# numbers to detect in norms)
# XXX this should be kept in sync with ./sieve/ecm/batch.cpp
# (output_batch)

sub usage {
    print <<EOF;
Usage: ./inspect-batch-file.pl [options]

Options understood:
    -batch <batch file> (mandatory) file to parse. Multiple files can be
    parsed in a single script call.

EOF
    die @_ if @_;
}

my @batchfiles=();

while (defined($_=shift @ARGV)) {
    /^-batch$/ && scalar @ARGV && do {
        push @batchfiles, shift @ARGV;
        next;
    };
    (/^--help$/ || /^-h$/) && do { usage; exit 0; };
    usage "ERROR: Unexpected argument: $_";
}

if (!scalar @batchfiles) {
    usage "ERROR: No -batchfile argument provided";
}

for my $batchfile (@batchfiles) {
    open F, "$batchfile" or die "$batchfile: $!";
    my $size = (stat $batchfile)[7];
    printf "$batchfile has size $size (0x%x)\n", $size;

    defined($_=<F>) or die;
    chomp($_);
    my $B = $_;
    defined($_=<F>) or die;
    chomp($_);
    my $L = $_;
    # B is 0 when that side is not sieved at all (lim = 0)
    my $lg = sub { my $x = shift; $x > 0 ? sprintf("2^%.1f", log($x)/log(2)) : "0"; };
    printf "batch file corresponds to B=$B (%s) and L=$L (%s)\n",
        $lg->($B), $lg->($L);
    defined($_=<F>) or die;
    chomp($_);
    my @coeffs = split('', $_);
    print "Coefficients of the corresponding polynomial (leading coefficient last):\n", @coeffs, "\n";
    close F;
}
