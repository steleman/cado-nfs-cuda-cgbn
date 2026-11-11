# An example strategy file for testing with one side.

a = { ECM-B12 50 5000 }
A = { PP1-65 150 4500 a a a }

10: { PM1 200 2000 }
1: { A }
2-4: { A[1:] ECM-TM12(12345) 200 2000 }
