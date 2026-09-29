"""BLOSUM62 + affine global Needleman-Wunsch (carried over unchanged)."""
import numpy as np

B62_ORDER = "ARNDCQEGHILKMFPSTWYVBZX*"
B62_TXT = """4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0 -2 -1 0 -4
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3 -1 0 -1 -4
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3 3 0 -1 -4
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3 4 1 -1 -4
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2 0 3 -1 -4
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2 1 4 -1 -4
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3 -1 -2 -1 -4
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3 0 0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1 -4 -3 -1 -4
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2 0 1 -1 -4
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2 -2 -1 -2 -4
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2 0 0 0 -4
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0 -1 -1 0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1 -3 -2 -1 -4
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4 -3 -2 -1 -4
-2 -1 3 4 -3 0 1 -1 0 -3 -4 0 -3 -3 -2 0 -1 -4 -3 -3 4 1 -1 -4
-1 0 0 1 -3 3 4 -2 0 -3 -3 1 -1 -3 -1 0 -1 -3 -2 -2 1 4 -1 -4
0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2 0 0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 1"""
B62 = {}
for i, row in enumerate(B62_TXT.strip().split("\n")):
    for j, v in enumerate(row.split()):
        B62[(B62_ORDER[i], B62_ORDER[j])] = int(v)
def sc(a, b):
    return B62.get((a, b), B62.get((b, a), -1))

def nw(s1, s2, gap_open=-11, gap_ext=-1):
    """global affine Needleman-Wunsch -> (aln1, aln2)"""
    n, m = len(s1), len(s2)
    NEG = -1e9
    M = np.full((n+1, m+1), NEG); Ix = np.full((n+1, m+1), NEG); Iy = np.full((n+1, m+1), NEG)
    tM = np.zeros((n+1, m+1), np.int8); tX = np.zeros((n+1, m+1), np.int8); tY = np.zeros((n+1, m+1), np.int8)
    M[0, 0] = 0
    for i in range(1, n+1): Ix[i, 0] = gap_open + (i-1)*gap_ext; tX[i, 0] = 1
    for j in range(1, m+1): Iy[0, j] = gap_open + (j-1)*gap_ext; tY[0, j] = 2
    for i in range(1, n+1):
        a = s1[i-1]
        for j in range(1, m+1):
            s = sc(a, s2[j-1])
            cands = (M[i-1, j-1], Ix[i-1, j-1], Iy[i-1, j-1])
            k = int(np.argmax(cands)); M[i, j] = cands[k] + s; tM[i, j] = k
            cands = (M[i-1, j] + gap_open, Ix[i-1, j] + gap_ext)
            k = int(np.argmax(cands)); Ix[i, j] = cands[k]; tX[i, j] = k
            cands = (M[i, j-1] + gap_open, Iy[i, j-1] + gap_ext)
            k = int(np.argmax(cands)); Iy[i, j] = cands[k]; tY[i, j] = k
    i, j = n, m
    state = int(np.argmax((M[n, m], Ix[n, m], Iy[n, m])))
    a1, a2 = [], []
    while i > 0 or j > 0:
        if state == 0:
            a1.append(s1[i-1]); a2.append(s2[j-1]); state = tM[i, j]; i -= 1; j -= 1
        elif state == 1:
            a1.append(s1[i-1]); a2.append("-"); state = 0 if tX[i, j] == 0 else 1; i -= 1
        else:
            a1.append("-"); a2.append(s2[j-1]); state = 0 if tY[i, j] == 0 else 2; j -= 1
        if i == 0 and j > 0: state = 2
        if j == 0 and i > 0: state = 1
    return "".join(reversed(a1)), "".join(reversed(a2))

