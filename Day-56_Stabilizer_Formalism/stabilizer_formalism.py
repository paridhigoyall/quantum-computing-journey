import numpy as np

# ============================================================
# DAY 56 — STABILIZER FORMALISM
#
# Goals:
#   1. Understand stabilizers
#   2. Verify Bell-state stabilizers
#   3. Demonstrate commutation / anticommutation
#   4. Generate error syndromes
#   5. Build the 3-qubit bit-flip code syndrome table
# ============================================================

np.set_printoptions(precision=6, suppress=True)


# ============================================================
# 1. BASIC PAULI MATRICES
# ============================================================

I = np.eye(2, dtype=complex)

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


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def kron_all(operators):
    """Kronecker product of multiple operators."""

    result = operators[0]

    for operator in operators[1:]:
        result = np.kron(result, operator)

    return result


def expectation(state, operator):
    """
    Calculate <psi|O|psi>.
    """

    return np.vdot(
        state,
        operator @ state
    )


def is_commuting(A, B):
    """
    Check whether AB = BA.
    """

    return np.allclose(
        A @ B,
        B @ A
    )


def is_anticommuting(A, B):
    """
    Check whether AB = -BA.
    """

    return np.allclose(
        A @ B,
        -B @ A
    )


def syndrome_bit(value):
    """
    Convert stabilizer eigenvalue into a syndrome bit.

        +1 -> 0
        -1 -> 1
    """

    if value >= 0:
        return 0

    return 1


# ============================================================
# 3. INTRODUCTION
# ============================================================

print("=" * 70)
print("DAY 56 — STABILIZER FORMALISM")
print("=" * 70)

print("""
A stabilizer S satisfies:

    S|psi> = |psi>

The state is therefore a +1 eigenstate
of the stabilizer.
""")


# ============================================================
# 4. BELL STATE
# ============================================================

zero = np.array([
    [1],
    [0]
], dtype=complex)

one = np.array([
    [0],
    [1]
], dtype=complex)


# |00>
zero_zero = np.kron(
    zero,
    zero
)

# |11>
one_one = np.kron(
    one,
    one
)


# Bell state:
#
# |Phi+> = (|00> + |11>) / sqrt(2)

bell_state = (
    zero_zero + one_one
) / np.sqrt(2)


print("\n" + "-" * 70)
print("1. BELL STATE")
print("-" * 70)

print("\n|Phi+> = (|00> + |11>) / sqrt(2)")

print("\nBell state vector:")
print(bell_state.flatten())

print(
    f"\nNormalization = "
    f"{np.linalg.norm(bell_state) ** 2:.6f}"
)


# ============================================================
# 5. TWO-QUBIT OPERATORS
# ============================================================

X0 = np.kron(X, I)
X1 = np.kron(I, X)

Y0 = np.kron(Y, I)
Y1 = np.kron(I, Y)

Z0 = np.kron(Z, I)
Z1 = np.kron(I, Z)


# ============================================================
# 6. BELL STATE STABILIZERS
# ============================================================

ZZ = Z0 @ Z1
XX = X0 @ X1


print("\n" + "-" * 70)
print("2. BELL STATE STABILIZERS")
print("-" * 70)

print("""
Candidate stabilizers:

    S1 = Z0 Z1
    S2 = X0 X1
""")


# Check ZZ
zz_value = np.real(
    expectation(
        bell_state,
        ZZ
    )
)

# Check XX
xx_value = np.real(
    expectation(
        bell_state,
        XX
    )
)

print(
    f"<Phi+| Z0Z1 |Phi+> = "
    f"{zz_value:.6f}"
)

print(
    f"<Phi+| X0X1 |Phi+> = "
    f"{xx_value:.6f}"
)


if abs(zz_value - 1) < 1e-10:
    print("\nSUCCESS: Z0Z1 stabilizes |Phi+>.")

if abs(xx_value - 1) < 1e-10:
    print("SUCCESS: X0X1 stabilizes |Phi+>.")


# ============================================================
# 7. VERIFY THE STABILIZER DIRECTLY
# ============================================================

print("\n" + "-" * 70)
print("3. DIRECT STABILIZER VERIFICATION")
print("-" * 70)

zz_state = ZZ @ bell_state
xx_state = XX @ bell_state


print("\nZZ|Phi+> =")
print(zz_state.flatten())

print("\n|Phi+> =")
print(bell_state.flatten())


if np.allclose(
    zz_state,
    bell_state
):
    print("\nZZ|Phi+> = |Phi+>")


print("\nXX|Phi+> =")
print(xx_state.flatten())

if np.allclose(
    xx_state,
    bell_state
):
    print("\nXX|Phi+> = |Phi+>")


# ============================================================
# 8. COMMUTATION
# ============================================================

print("\n" + "-" * 70)
print("4. COMMUTATION AND ANTICOMMUTATION")
print("-" * 70)


print("\nPauli relationship:")

print("\nX Z =")
print(X @ Z)

print("\nZ X =")
print(Z @ X)


if is_anticommuting(X, Z):

    print("\nSUCCESS: X and Z ANTICOMMUTE.")

    print("Therefore:")
    print("    XZ = -ZX")


print("\nX X =")
print(X @ X)

if is_commuting(X, X):

    print("\nSUCCESS: X and X COMMUTE.")


# ============================================================
# 9. APPLY X ERROR TO BELL STATE
# ============================================================

print("\n" + "-" * 70)
print("5. X ERROR AND SYNDROME")
print("-" * 70)

print("""
Initially:

    Z0Z1 = +1
    X0X1 = +1

Now apply X error to qubit 0.
""")


error_state = X0 @ bell_state


# Measure stabilizers after error

zz_after_error = np.real(
    expectation(
        error_state,
        ZZ
    )
)

xx_after_error = np.real(
    expectation(
        error_state,
        XX
    )
)


print(
    f"\nAfter X0 error:"
)

print(
    f"Z0Z1 = {zz_after_error:.6f}"
)

print(
    f"X0X1 = {xx_after_error:.6f}"
)

print("\nSyndrome:")

print(
    f"Z0Z1 syndrome bit = "
    f"{syndrome_bit(zz_after_error)}"
)

print(
    f"X0X1 syndrome bit = "
    f"{syndrome_bit(xx_after_error)}"
)


# ============================================================
# 10. THREE-QUBIT BIT-FLIP CODE
# ============================================================

print("\n" + "=" * 70)
print("6. THREE-QUBIT BIT-FLIP CODE")
print("=" * 70)


# Logical state
alpha = 1 / np.sqrt(3)
beta = np.sqrt(2 / 3)


# |000>
zero_zero_zero = kron_all([
    zero.flatten(),
    zero.flatten(),
    zero.flatten()
])

# |111>
one_one_one = kron_all([
    one.flatten(),
    one.flatten(),
    one.flatten()
])


# Encoded state:
#
# |psi_L> = alpha|000> + beta|111>

encoded_state = (
    alpha * zero_zero_zero
    + beta * one_one_one
)


print("\nLogical state:")

print(
    "|psi> = alpha|0> + beta|1>"
)

print(f"\nalpha = {alpha:.6f}")
print(f"beta  = {beta:.6f}")


print(
    f"\nEncoded state normalization = "
    f"{np.linalg.norm(encoded_state) ** 2:.6f}"
)


# ============================================================
# 11. THREE-QUBIT OPERATORS
# ============================================================

X0_3 = kron_all([
    X,
    I,
    I
])

X1_3 = kron_all([
    I,
    X,
    I
])

X2_3 = kron_all([
    I,
    I,
    X
])


Z0_3 = kron_all([
    Z,
    I,
    I
])

Z1_3 = kron_all([
    I,
    Z,
    I
])

Z2_3 = kron_all([
    I,
    I,
    Z
])


# ============================================================
# 12. BIT-FLIP CODE STABILIZERS
# ============================================================

S1 = Z0_3 @ Z1_3

S2 = Z1_3 @ Z2_3


print("\nBit-flip code stabilizers:")

print("\nS1 = Z0 Z1")
print("S2 = Z1 Z2")


# ============================================================
# 13. FUNCTION TO CALCULATE THREE-QUBIT SYNDROME
# ============================================================

def get_3qubit_syndrome(state):

    s1 = np.real(
        expectation(
            state,
            S1
        )
    )

    s2 = np.real(
        expectation(
            state,
            S2
        )
    )

    return (
        syndrome_bit(s1),
        syndrome_bit(s2)
    )


# ============================================================
# 14. ERROR IDENTIFICATION
# ============================================================

def identify_3qubit_error(syndrome):

    mapping = {

        (0, 0): "No error",

        (1, 0): "X error on qubit 0",

        (1, 1): "X error on qubit 1",

        (0, 1): "X error on qubit 2"
    }

    return mapping[syndrome]


# ============================================================
# 15. TEST NO ERROR
# ============================================================

print("\n" + "-" * 70)
print("7. NO ERROR")
print("-" * 70)

no_error_state = encoded_state

syndrome = get_3qubit_syndrome(
    no_error_state
)

print(
    f"\nSyndrome = {syndrome}"
)

print(
    f"Detected = "
    f"{identify_3qubit_error(syndrome)}"
)


# ============================================================
# 16. TEST X0
# ============================================================

print("\n" + "-" * 70)
print("8. X ERROR ON QUBIT 0")
print("-" * 70)

state_x0 = X0_3 @ encoded_state

syndrome_x0 = get_3qubit_syndrome(
    state_x0
)

print(
    f"\nSyndrome = {syndrome_x0}"
)

print(
    f"Detected = "
    f"{identify_3qubit_error(syndrome_x0)}"
)


# ============================================================
# 17. TEST X1
# ============================================================

print("\n" + "-" * 70)
print("9. X ERROR ON QUBIT 1")
print("-" * 70)

state_x1 = X1_3 @ encoded_state

syndrome_x1 = get_3qubit_syndrome(
    state_x1
)

print(
    f"\nSyndrome = {syndrome_x1}"
)

print(
    f"Detected = "
    f"{identify_3qubit_error(syndrome_x1)}"
)


# ============================================================
# 18. TEST X2
# ============================================================

print("\n" + "-" * 70)
print("10. X ERROR ON QUBIT 2")
print("-" * 70)

state_x2 = X2_3 @ encoded_state

syndrome_x2 = get_3qubit_syndrome(
    state_x2
)

print(
    f"\nSyndrome = {syndrome_x2}"
)

print(
    f"Detected = "
    f"{identify_3qubit_error(syndrome_x2)}"
)


# ============================================================
# 19. AUTOMATIC SYNDROME TABLE
# ============================================================

print("\n" + "=" * 70)
print("11. AUTOMATIC SYNDROME TABLE")
print("=" * 70)

errors = {

    "No error":
        np.eye(8, dtype=complex),

    "X0":
        X0_3,

    "X1":
        X1_3,

    "X2":
        X2_3
}


print(
    "\nError\t\tSyndrome\tDetected"
)

print(
    "-" * 60
)

for error_name, error_operator in errors.items():

    corrupted_state = (
        error_operator @ encoded_state
    )

    syndrome = get_3qubit_syndrome(
        corrupted_state
    )

    detected = identify_3qubit_error(
        syndrome
    )

    print(
        f"{error_name:<12}"
        f"{str(syndrome):<16}"
        f"{detected}"
    )


# ============================================================
# 20. EXPLAIN ANTICOMMUTATION
# ============================================================

print("\n" + "=" * 70)
print("12. WHY DOES THE SYNDROME CHANGE?")
print("=" * 70)

print("""
Suppose:

    S|psi> = +|psi>

and an error E occurs.

If:

    SE = ES

then:

    S(E|psi>) = +E|psi>

The stabilizer stays +1.

But if:

    SE = -ES

then:

    S(E|psi>) = -E|psi>

The stabilizer changes from:

    +1 -> -1

That change becomes a syndrome bit:

    +1 -> 0
    -1 -> 1
""")


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("DAY 56 SUMMARY")
print("=" * 70)

print("""
STABILIZER:

    S|psi> = |psi>

means S stabilizes the state.

ERROR DETECTION:

    Commuting error:
        stabilizer remains +1

    Anticommuting error:
        stabilizer changes +1 -> -1

SYNDROME:

    +1 -> 0
    -1 -> 1

3-QUBIT BIT-FLIP CODE:

    S1 = Z0 Z1
    S2 = Z1 Z2

Syndromes:

    00 -> No error
    10 -> X0
    11 -> X1
    01 -> X2

KEY IDEA:

    The syndrome is an error fingerprint.

This is the mathematical foundation behind
stabilizer-based quantum error correction.
""")

print("=" * 70)
print("DAY 56 COMPLETE")
print("=" * 70)