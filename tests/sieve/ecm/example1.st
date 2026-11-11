# An example strategy file, which exercises corner cases of the parser.
# The longer one is ../F9.st. Newlines are not significant.

a = { ECM-B12 50 5000 }     # a fragment without a side serves either
b={ECM-B12 50 5000}
A = { PP1-65 150 4500 a a a }
B = {
    PP1-65 150 4500
    a b
}
C = { 1: PM1 200 2000 }     # a side inside a fragment stays inside

10 20: { PM1 200 2000 }
1 45: { A }
1 46: { B }                 # comment at end of line
1 47: { 0: A[1:3] 1: B[:1] C B[1:] }
2 47: {
    ECM-B12 200 2000
    ECM-B12(12345) 200 2000
    ECM-B12(6789) 200 2000
    ECM-M12 200 2000
    ECM-M12(12345) 200 2000
    ECM-M16 200 2000
    ECM-TM12(12345) 200 2000
}
3-5 40-42: { 1: PM1 315 2205 STOP MPQS 0: PM1 315 2205 }
4 41: { }                   # rejects one cell of the range above
