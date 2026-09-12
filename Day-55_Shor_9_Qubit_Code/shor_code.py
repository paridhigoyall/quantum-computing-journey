import numpy as np

# ============================================================
# DAY 55 — SHOR'S 9-QUBIT QUANTUM ERROR-CORRECTING CODE
#
# Goal:
#   1. Encode one logical qubit into 9 physical qubits
#   2. Create the Shor code logical states
#   3. Apply X, Z and Y errors
#   4. Detect the error using stabilizer syndromes
#   5. Correct a single-qubit error
#   6. Check fidelity
#   7. Demonstrate the limitation with two errors
# ============================================================

np.set_printoptions(precision=6, suppress=True)


# ============================================================
# 1. BASIC MATRICES
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

H = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
], dtype=complex)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def kron_all(operators):
    """Kronecker product of a list of matrices."""
    result = operators[0]

    for operator in operators[1:]:
        result = np.kron(result, operator)

    return result


def n_qubit_operator(single_operator, qubit, n=9):
    """
    Put a single-qubit operator on a specific qubit.

    Qubits are numbered:
        0, 1, 2, ..., 8
    """

    operators = []

    for q in range(n):
        if q == qubit:
            operators.append(single_operator)
        else:
            operators.append(I)

    return kron_all(operators)


def expectation(state, operator):
    """Calculate <psi|O|psi>."""
    return np.vdot(state, operator @ state)


def same_state_up_to_global_phase(state1, state2, tolerance=1e-10):
    """
    Check whether two states are equal up to global phase.
    """

    overlap = np.vdot(state1, state2)

    if abs(overlap) < tolerance:
        return False

    phase = overlap / abs(overlap)

    return np.allclose(
        state1,
        phase * state2,
        atol=tolerance
    )


def fidelity(state1, state2):
    """F = |<psi|phi>|^2."""
    return abs(np.vdot(state1, state2)) ** 2


# ============================================================
# 3. LOGICAL QUBIT
# ============================================================

alpha = 1 / np.sqrt(3)
beta = np.sqrt(2 / 3)

logical_state = np.array([
    alpha,
    beta
], dtype=complex)


print("=" * 75)
print("DAY 55 — SHOR'S 9-QUBIT QUANTUM ERROR-CORRECTING CODE")
print("=" * 75)

print("\nLogical state:")
print("|psi> = alpha|0> + beta|1>")

print(f"\nalpha = {alpha:.6f}")
print(f"beta  = {beta:.6f}")

print(
    f"\nLogical state normalization = "
    f"{np.linalg.norm(logical_state) ** 2:.6f}"
)


# ============================================================
# 4. THREE-QUBIT REPETITION STATES
# ============================================================

zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
minus = np.array([1, -1], dtype=complex) / np.sqrt(2)


# |000>
zero_zero_zero = np.kron(
    np.kron(zero, zero),
    zero
)

# |111>
one_one_one = np.kron(
    np.kron(one, one),
    one
)

# |+++>
plus_plus_plus = np.kron(
    np.kron(plus, plus),
    plus
)

# |--->
minus_minus_minus = np.kron(
    np.kron(minus, minus),
    minus
)


# ============================================================
# 5. SHOR CODE LOGICAL STATES
# ============================================================
#
# |0_L> =
#       1/(2sqrt(2))
#       (|000> + |111>)
#       (|000> + |111>)
#       (|000> + |111>)
#
# |1_L> =
#       1/(2sqrt(2))
#       (|000> - |111>)
#       (|000> - |111>)
#       (|000> - |111>)
#
# A convenient way to construct them is:
#
# |0_L> = |+++> + |--->
#          with proper normalization
#
# |1_L> = |+++> - |--->
#          with proper normalization
#
# More explicitly:
#
# |0_L> = (|000>+|111>)^3 / (2sqrt(2))
# |1_L> = (|000>-|111>)^3 / (2sqrt(2))
#
# ============================================================

ghz_plus = (
    zero_zero_zero + one_one_one
) / np.sqrt(2)

ghz_minus = (
    zero_zero_zero - one_one_one
) / np.sqrt(2)


# |0_L> = GHZ+ GHZ+ GHZ+
logical_zero = kron_all([
    ghz_plus,
    ghz_plus,
    ghz_plus
])

# |1_L> = GHZ- GHZ- GHZ-
logical_one = kron_all([
    ghz_minus,
    ghz_minus,
    ghz_minus
])


# ============================================================
# 6. VERIFY LOGICAL STATE NORMALIZATION
# ============================================================

print("\n" + "-" * 75)
print("SHOR CODE ENCODING")
print("-" * 75)

print("\n|0_L> is encoded into 9 physical qubits.")
print("|1_L> is encoded into 9 physical qubits.")

print(
    f"\n||0_L||² = "
    f"{np.linalg.norm(logical_zero) ** 2:.6f}"
)

print(
    f"||1_L||² = "
    f"{np.linalg.norm(logical_one) ** 2:.6f}"
)


# ============================================================
# 7. ENCODE ARBITRARY LOGICAL STATE
# ============================================================

encoded_state = (
    alpha * logical_zero
    + beta * logical_one
)

print(
    f"\nEncoded state normalization = "
    f"{np.linalg.norm(encoded_state) ** 2:.6f}"
)


# ============================================================
# 8. BUILD 9-QUBIT PAULI OPERATORS
# ============================================================

X_ops = []
Y_ops = []
Z_ops = []

for q in range(9):
    X_ops.append(n_qubit_operator(X, q))
    Y_ops.append(n_qubit_operator(Y, q))
    Z_ops.append(n_qubit_operator(Z, q))


# ============================================================
# 9. SHOR CODE STABILIZERS
# ============================================================
#
# First six stabilizers detect bit flips inside each group:
#
# Z0 Z1
# Z1 Z2
# Z3 Z4
# Z4 Z5
# Z6 Z7
# Z7 Z8
#
# The remaining stabilizers detect phase information:
#
# X0 X1 X2 X3 X4 X5
# X3 X4 X5 X6 X7 X8
#
# ============================================================

Z0Z1 = Z_ops[0] @ Z_ops[1]
Z1Z2 = Z_ops[1] @ Z_ops[2]

Z3Z4 = Z_ops[3] @ Z_ops[4]
Z4Z5 = Z_ops[4] @ Z_ops[5]

Z6Z7 = Z_ops[6] @ Z_ops[7]
Z7Z8 = Z_ops[7] @ Z_ops[8]

X012345 = (
    X_ops[0]
    @ X_ops[1]
    @ X_ops[2]
    @ X_ops[3]
    @ X_ops[4]
    @ X_ops[5]
)

X345678 = (
    X_ops[3]
    @ X_ops[4]
    @ X_ops[5]
    @ X_ops[6]
    @ X_ops[7]
    @ X_ops[8]
)


stabilizers = [
    Z0Z1,
    Z1Z2,
    Z3Z4,
    Z4Z5,
    Z6Z7,
    Z7Z8,
    X012345,
    X345678
]


# ============================================================
# 10. SYNDROME CALCULATION
# ============================================================

def get_syndrome(state):

    syndrome = []

    for stabilizer in stabilizers:

        value = np.real(
            expectation(state, stabilizer)
        )

        if value >= 0:
            syndrome.append(0)
        else:
            syndrome.append(1)

    return tuple(syndrome)


# ============================================================
# 11. IDENTIFY ERROR
# ============================================================

def identify_error(syndrome):

    # --------------------------------------------------------
    # No error
    # --------------------------------------------------------

    if syndrome == (0, 0, 0, 0, 0, 0, 0, 0):
        return "No error"

    # --------------------------------------------------------
    # Bit-flip errors
    # --------------------------------------------------------

    bit_flip_map = {

        (1, 0, 0, 0, 0, 0, 0, 0): "X on qubit 0",

        (1, 1, 0, 0, 0, 0, 0, 0): "X on qubit 1",

        (0, 1, 0, 0, 0, 0, 0, 0): "X on qubit 2",

        (0, 0, 1, 0, 0, 0, 0, 0): "X on qubit 3",

        (0, 0, 1, 1, 0, 0, 0, 0): "X on qubit 4",

        (0, 0, 0, 1, 0, 0, 0, 0): "X on qubit 5",

        (0, 0, 0, 0, 1, 0, 0, 0): "X on qubit 6",

        (0, 0, 0, 0, 1, 1, 0, 0): "X on qubit 7",

        (0, 0, 0, 0, 0, 1, 0, 0): "X on qubit 8",
    }

    if syndrome in bit_flip_map:
        return bit_flip_map[syndrome]

    # --------------------------------------------------------
    # Phase-flip errors
    #
    # The final two stabilizers identify which group has
    # a phase error.
    #
    # Within a group, Z errors have the same phase syndrome.
    # --------------------------------------------------------

    phase_group_map = {

        (0, 0, 0, 0, 0, 0, 1, 0):
            "Phase error in group 1",

        (0, 0, 0, 0, 0, 0, 1, 1):
            "Phase error in group 2",

        (0, 0, 0, 0, 0, 0, 0, 1):
            "Phase error in group 3",
    }

    if syndrome in phase_group_map:
        return phase_group_map[syndrome]

    return "Complex / unrecognized error"


# ============================================================
# 12. CORRECTION
# ============================================================

def correct_error(state, syndrome):

    # --------------------------------------------------------
    # No error
    # --------------------------------------------------------

    if syndrome == (0, 0, 0, 0, 0, 0, 0, 0):

        print("No correction required.")

        return state


    # --------------------------------------------------------
    # Bit-flip syndrome correction
    # --------------------------------------------------------

    bit_flip_corrections = {

        (1, 0, 0, 0, 0, 0, 0, 0): 0,
        (1, 1, 0, 0, 0, 0, 0, 0): 1,
        (0, 1, 0, 0, 0, 0, 0, 0): 2,

        (0, 0, 1, 0, 0, 0, 0, 0): 3,
        (0, 0, 1, 1, 0, 0, 0, 0): 4,
        (0, 0, 0, 1, 0, 0, 0, 0): 5,

        (0, 0, 0, 0, 1, 0, 0, 0): 6,
        (0, 0, 0, 0, 1, 1, 0, 0): 7,
        (0, 0, 0, 0, 0, 1, 0, 0): 8,
    }

    if syndrome in bit_flip_corrections:

        qubit = bit_flip_corrections[syndrome]

        print(
            f"Applying X correction to qubit {qubit}..."
        )

        return X_ops[qubit] @ state


    # --------------------------------------------------------
    # Phase-flip correction
    #
    # We choose the first qubit of the affected group.
    # --------------------------------------------------------

    phase_corrections = {

        (0, 0, 0, 0, 0, 0, 1, 0): 0,

        (0, 0, 0, 0, 0, 0, 1, 1): 3,

        (0, 0, 0, 0, 0, 0, 0, 1): 6,
    }

    if syndrome in phase_corrections:

        qubit = phase_corrections[syndrome]

        print(
            f"Applying Z correction to qubit {qubit}..."
        )

        return Z_ops[qubit] @ state


    print("Unable to determine a correction.")

    return state


# ============================================================
# 13. TEST FUNCTION
# ============================================================

def test_error(error_name, error_operator):

    print("\n" + "=" * 75)
    print(f"TEST: {error_name}")
    print("=" * 75)

    # Apply error
    corrupted_state = (
        error_operator @ encoded_state
    )

    # Calculate syndrome
    syndrome = get_syndrome(
        corrupted_state
    )

    print(f"\nSyndrome = {syndrome}")

    # Identify error
    detected = identify_error(
        syndrome
    )

    print(f"Detected = {detected}")

    # Correct
    corrected_state = correct_error(
        corrupted_state,
        syndrome
    )

    # Fidelity with original encoded state
    F = fidelity(
        encoded_state,
        corrected_state
    )

    print(f"\nFidelity = {F:.6f}")

    if abs(F - 1.0) < 1e-10:

        print(
            "SUCCESS: Single-qubit error corrected."
        )

    else:

        print(
            "Correction was not perfect."
        )

    return corrected_state


# ============================================================
# 14. TEST NO ERROR
# ============================================================

test_error(
    "NO ERROR",
    np.eye(512, dtype=complex)
)


# ============================================================
# 15. TEST X ERRORS
# ============================================================

test_error(
    "BIT FLIP X on qubit 0",
    X_ops[0]
)

test_error(
    "BIT FLIP X on qubit 4",
    X_ops[4]
)

test_error(
    "BIT FLIP X on qubit 8",
    X_ops[8]
)


# ============================================================
# 16. TEST Z ERRORS
# ============================================================

test_error(
    "PHASE FLIP Z on qubit 0",
    Z_ops[0]
)

test_error(
    "PHASE FLIP Z on qubit 4",
    Z_ops[4]
)

test_error(
    "PHASE FLIP Z on qubit 8",
    Z_ops[8]
)


# ============================================================
# 17. TEST Y ERRORS
# ============================================================

test_error(
    "Y ERROR on qubit 0",
    Y_ops[0]
)

test_error(
    "Y ERROR on qubit 4",
    Y_ops[4]
)

test_error(
    "Y ERROR on qubit 8",
    Y_ops[8]
)


# ============================================================
# 18. TWO-ERROR LIMITATION
# ============================================================

print("\n" + "=" * 75)
print("TWO-ERROR LIMITATION TEST")
print("=" * 75)

print("\nApplying X errors to qubit 0 and qubit 1.")

two_error_operator = (
    X_ops[0] @ X_ops[1]
)

two_error_state = (
    two_error_operator @ encoded_state
)

two_error_syndrome = get_syndrome(
    two_error_state
)

print(
    f"\nSyndrome = {two_error_syndrome}"
)

print(
    f"Decoder interpretation = "
    f"{identify_error(two_error_syndrome)}"
)

two_error_corrected = correct_error(
    two_error_state,
    two_error_syndrome
)

two_error_fidelity = fidelity(
    encoded_state,
    two_error_corrected
)

print(
    f"\nFidelity after attempted correction = "
    f"{two_error_fidelity:.6f}"
)

print(
    "\nThis demonstrates that the simple Shor code "
    "is designed to correct ONE arbitrary single-qubit error."
)


# ============================================================
# 19. ERROR SUMMARY TABLE
# ============================================================

print("\n" + "=" * 75)
print("SHOR CODE ERROR-CORRECTION SUMMARY")
print("=" * 75)

print("""
Error type              Correctable?
-------------------------------------
I   (no error)           YES
X   (bit flip)           YES
Z   (phase flip)         YES
Y   (bit + phase)        YES

Two-qubit errors         NOT GUARANTEED
""")


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("=" * 75)
print("DAY 55 SUMMARY")
print("=" * 75)

print("""
Shor's 9-qubit code:

    1 logical qubit
            |
            v
    9 physical qubits

The code combines:

    Bit-flip protection
            +
    Phase-flip protection

Pauli errors:

    I = no error
    X = bit flip
    Z = phase flip
    Y = bit + phase flip

Important relation:

    Y = i X Z

Capability:

    Corrects any ONE single-qubit Pauli error.

    X_i  -> correctable
    Y_i  -> correctable
    Z_i  -> correctable

The important limitation:

    Multiple simultaneous errors are not
    generally guaranteed to be correctable.

This is the basic idea behind Shor's
9-qubit quantum error-correcting code.
""")

print("=" * 75)
print("DAY 55 COMPLETE")
print("=" * 75)