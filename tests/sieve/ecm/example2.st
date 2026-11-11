# An example strategy file for testing with more than two sides.

a = { ECM-B12 50 5000 }
A = { PP1-65 150 4500 a a a }

10 20 25: { PM1 200 2000 }
1 45 10: { A }
1 46 30: { 2: A[:2] 1: a }
2 47 25: {
    ECM-B12 200 2000
    ECM-B12(12345) 200 2000
    1: ECM-M12 200 2000
    2: ECM-M16 200 2000
}
