#!/usr/bin/env python3

# Generates gnuplot commands to plot the sieve region in the a,b-plane,
# and the relations found. One .png file is generated for each special-q
# in the siever output file. Takes one command line parameter with the
# siever output file; if not specified, reads from stdin

import sys
import re

# the lines of las output matched below are listed in
# sieve/las-special-markers.hpp
pattern1 = re.compile(r"# Sieving side-\d+ q=(\d+)(?:=[\d*]+)?; rho=(\d+);"
                      r" a0=(-?\d+); b0=(-?\d+); a1=(-?\d+); b1=(-?\d+);"
                      r" I=(\d+); J=(\d+);")
pattern2 = re.compile(r"# Time for side-\d+ q=")
pattern3 = re.compile(r"(-?\d+),(\d+):")


def print_gnuplot(q, r, a0, b0, a1, b1, i, j, rels, outputfilename):
    # First corner: (-I,0) -> (-I * a0, -I * b0)
    c1x = -i // 2 * a0
    c1y = -i // 2 * b0
    # Second corner: (-I,J) -> (-I * a0 + J * a1, -I * b0 + J * b1)
    c2x = -i // 2 * a0 + j * a1
    c2y = -i // 2 * b0 + j * b1
    # Third corner: (I,J) -> (I * a0 + J * a1, I * b0 + J * b1)
    c3x = i // 2 * a0 + j * a1
    c3y = i // 2 * b0 + j * b1
    # Fourth corner: (I,0) -> (I * a0, I * b0)
    c4x = i // 2 * a0
    c4y = i // 2 * b0

    xmin = min(c1x, c2x, c3x, c4x)
    xmax = max(c1x, c2x, c3x, c4x)
    ymin = min(c1y, c2y, c3y, c4y)
    ymax = max(c1y, c2y, c3y, c4y)

    if ymin < 0:
        xmin = min(xmin, -xmax)
        xmax = max(xmax, -xmin)
        ymax = max(ymax, -ymin)

    print('set terminal png size 1200,900 enhanced font "Helvetica,20"')
    print("set output 'sieve_region.%d.%d.png'" % (q, r))

    print("set xrange [%d:%d]" % (xmin, xmax))
    print("set yrange [%d:%d]" % (ymin, ymax))
    print("set xzeroaxis")
    print("set yzeroaxis")
    print('set title "q=%d, r=%d, I=%d, J=%d, yield: %d"'
          % (q, r, i, j, rels))
    print("set object 1 polygon"
          " from %d,%d to %d,%d to %d,%d to %d,%d to %d,%d"
          % (c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y, c1x, c1y))
    print('set label 1 "(-I/2,0)" at %d,%d' % (c1x, c1y))
    print('set label 2 "(-I/2,J)" at %d,%d' % (c2x, c2y))
    print('set label 3 "(I/2,J)" at %d,%d' % (c3x, c3y))
    print('set label 4 "(I/2,0)" at %d,%d' % (c4x, c4y))

    print('plot "%s"' % outputfilename)


def main(inputfile):
    outputfile = None
    for line in inputfile:
        match = pattern1.match(line)
        if match:
            (q, r, a0, b0, a1, b1, i, j) = map(int, match.groups())
            outputfilename = "relations_ab.%d.%d" % (q, r)
            outputfile = open(outputfilename, "w")
            rels = 0
            continue
        if pattern2.match(line) and outputfile:
            outputfile.close()
            outputfile = None
            print_gnuplot(q, r, a0, b0, a1, b1, i, j, rels, outputfilename)
            continue
        match = pattern3.match(line)
        if match and outputfile:
            (a, b) = map(int, match.groups())
            outputfile.write("%d %d\n" % (a, b))
            rels += 1

    if outputfile:
        outputfile.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            main(f)
    else:
        main(sys.stdin)
