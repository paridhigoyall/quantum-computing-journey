import numpy as np

# ============================================================
# DAY 54 — PAULI ERROR CONNECTION
#
# Goal:
# 1. Verify H X H = Z
# 2. Verify H Z H = X
# 3. Verify Y = i X Z
# 4. Demonstrate the action of X, Y, Z
# 5. Connect bit-flip and phase-flip error correction
# ============================================================

np.set_printoptions(precision=6, suppress=True)


# ------------------------------------------------------------
# 1. BASIC QUANTUM OPERATORS
# ------------------------------------------------------------

I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

H = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
], dtype=complex)


# ------------------------------------------------------------
# 2. HELPER FUNCTION
# ------------------------------------------------------------

def print_matrix(name, matrix):
    print(f"\n{name}:")
    print(matrix)


def matrices_equal(A, B, tolerance=1e-10):
    return np.allclose(A, B, atol=tolerance)


# ------------------------------------------------------------
# 3. DISPLAY BASIC OPERATORS
# ------------------------------------------------------------

print("=" * 65)
print("DAY 54 — PAULI ERROR CONNECTION")
print("=" * 65)

print_matrix("X (Bit-Flip)", X)
print_matrix("Y (Bit + Phase Flip)", Y)
print_matrix("Z (Phase-Flip)", Z)
print_matrix("H (Hadamard)", H)


# ------------------------------------------------------------
# 4. VERIFY H X H = Z
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 1 — H X H = Z")
print("-" * 65)

HXH = H @ X @ H

print("\nH X H =")
print(HXH)

print("\nZ =")
print(Z)

if matrices_equal(HXH, Z):
    print("\nSUCCESS: H X H = Z")
else:
    print("\nERROR: H X H != Z")


# ------------------------------------------------------------
# 5. VERIFY H Z H = X
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 2 — H Z H = X")
print("-" * 65)

HZH = H @ Z @ H

print("\nH Z H =")
print(HZH)

print("\nX =")
print(X)

if matrices_equal(HZH, X):
    print("\nSUCCESS: H Z H = X")
else:
    print("\nERROR: H Z H != X")


# ------------------------------------------------------------
# 6. VERIFY Y = i X Z
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 3 — Y = i X Z")
print("-" * 65)

iXZ = 1j * X @ Z

print("\ni X Z =")
print(iXZ)

print("\nY =")
print(Y)

if matrices_equal(iXZ, Y):
    print("\nSUCCESS: Y = i X Z")
else:
    print("\nERROR: Y != i X Z")


# ------------------------------------------------------------
# 7. BASIS STATES
# ------------------------------------------------------------

zero = np.array([
    [1],
    [0]
], dtype=complex)

one = np.array([
    [0],
    [1]
], dtype=complex)

plus = np.array([
    [1],
    [1]
], dtype=complex) / np.sqrt(2)

minus = np.array([
    [1],
    [-1]
], dtype=complex) / np.sqrt(2)


# ------------------------------------------------------------
# 8. SHOW BIT-FLIP
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 4 — BIT-FLIP")
print("-" * 65)

print("\nX|0> =")
print(X @ zero)

print("\nX|1> =")
print(X @ one)

print("\nTherefore:")

print("|0> --X--> |1>")
print("|1> --X--> |0>")


# ------------------------------------------------------------
# 9. SHOW PHASE-FLIP
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 5 — PHASE-FLIP")
print("-" * 65)

print("\nZ|0> =")
print(Z @ zero)

print("\nZ|1> =")
print(Z @ one)

print("\nTherefore:")

print("|0> --Z-->  |0>")
print("|1> --Z--> -|1>")


# ------------------------------------------------------------
# 10. PHASE FLIP IN X BASIS
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 6 — PHASE FLIP IN X BASIS")
print("-" * 65)

print("\n|+> =")
print(plus)

print("\n|-> =")
print(minus)

print("\nZ|+> =")
print(Z @ plus)

print("\n|-> =")
print(minus)

if matrices_equal(Z @ plus, minus):
    print("\nSUCCESS: Z|+> = |->")
else:
    print("\nERROR")


# ------------------------------------------------------------
# 11. HADAMARD CHANGES THE BASIS
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 7 — HADAMARD BASIS CHANGE")
print("-" * 65)

print("\nH|0> =")
print(H @ zero)

print("\n|+> =")
print(plus)

if matrices_equal(H @ zero, plus):
    print("\nSUCCESS: H|0> = |+>")
else:
    print("\nERROR")


print("\nH|1> =")
print(H @ one)

print("\n|-> =")
print(minus)

if matrices_equal(H @ one, minus):
    print("\nSUCCESS: H|1> = |->")
else:
    print("\nERROR")


# ------------------------------------------------------------
# 12. DEMONSTRATE ERROR TRANSFORMATION
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 8 — ERROR TRANSFORMATION")
print("-" * 65)

print("\nThe Hadamard gate transforms Pauli errors as:")

print("\nH X H = Z")
print("H Z H = X")

print("\nTherefore:")

print("Bit-flip X <----H----> Phase-flip Z")


# ------------------------------------------------------------
# 13. CHECK PAULI SQUARED
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("EXPERIMENT 9 — PAULI OPERATORS SQUARED")
print("-" * 65)

print("\nX² =")
print(X @ X)

print("\nY² =")
print(Y @ Y)

print("\nZ² =")
print(Z @ Z)

print("\nAll Pauli operators satisfy:")

print("X² = Y² = Z² = I")


# ------------------------------------------------------------
# 14. ERROR CORRECTION CONNECTION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("ERROR CORRECTION CONNECTION")
print("=" * 65)

print("""
DAY 52 — BIT-FLIP CODE

Error:
    X

Encoding:
    |0> -> |000>
    |1> -> |111>

Syndrome operators:
    Z0 Z1
    Z1 Z2


DAY 53 — PHASE-FLIP CODE

Error:
    Z

Encoding:
    |0> -> |+++>
    |1> -> |--->

Syndrome operators:
    X0 X1
    X1 X2
""")


# ------------------------------------------------------------
# 15. PAULI ERROR TABLE
# ------------------------------------------------------------

print("=" * 65)
print("PAULI ERROR TABLE")
print("=" * 65)

print("""
+-------+-------------------------+--------------------------+
| Error | Meaning                 | Code                    |
+-------+-------------------------+--------------------------+
|   I   | No error                | No correction           |
|   X   | Bit flip                | Bit-flip code           |
|   Z   | Phase flip              | Phase-flip code         |
|   Y   | Bit + phase flip        | Needs combined code    |
+-------+-------------------------+--------------------------+
""")


# ------------------------------------------------------------
# 16. WHY Y IS IMPORTANT
# ------------------------------------------------------------

print("=" * 65)
print("WHY Y IS IMPORTANT")
print("=" * 65)

print("""
Y = i X Z

So a Y error contains both:

    Bit-flip component
        +
    Phase-flip component

Therefore, correcting only X errors
or only Z errors is not enough.

For a general single-qubit error,
we need to handle:

    I, X, Y, Z
""")


# ------------------------------------------------------------
# 17. FINAL SUMMARY
# ------------------------------------------------------------

print("=" * 65)
print("DAY 54 SUMMARY")
print("=" * 65)

print("""
1. Hadamard changes the error basis.

       H X H = Z
       H Z H = X

2. Bit flip and phase flip are closely related.

       X <----H----> Z

3. Y combines both types of errors.

       Y = i X Z

4. Day 52:
       Bit-flip error correction.

5. Day 53:
       Phase-flip error correction.

6. The next goal:
       Correct arbitrary single-qubit errors.

       I, X, Y, Z
""")

print("=" * 65)
print("DAY 54 COMPLETE")
print("=" * 65)