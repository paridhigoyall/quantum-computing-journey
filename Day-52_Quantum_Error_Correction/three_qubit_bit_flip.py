import numpy as np


# ============================================================
# DAY 52
# 3-QUBIT BIT-FLIP QUANTUM ERROR-CORRECTION CODE
#
# Goal:
#   Encode one logical qubit into three physical qubits.
#   Introduce a bit-flip error.
#   Measure the error syndrome.
#   Identify the corrupted qubit.
#   Correct the error.
#
# Encoding:
#   |0> -> |000>
#   |1> -> |111>
#
# Logical state:
#   |psi> = alpha|0> + beta|1>
#
# Encoded state:
#   |psi_L> = alpha|000> + beta|111>
#
# Syndrome checks:
#   S1 = Z0 Z1
#   S2 = Z1 Z2
# ============================================================


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


# ============================================================
# 2. THREE-QUBIT OPERATORS
# ============================================================

I3 = np.kron(
    np.kron(I, I),
    I
)

X0 = np.kron(
    np.kron(X, I),
    I
)

X1 = np.kron(
    np.kron(I, X),
    I
)

X2 = np.kron(
    np.kron(I, I),
    X
)


Z0 = np.kron(
    np.kron(Z, I),
    I
)

Z1 = np.kron(
    np.kron(I, Z),
    I
)

Z2 = np.kron(
    np.kron(I, I),
    Z
)


# ============================================================
# 3. SYNDROME OPERATORS
#
# S1 = Z0 Z1
# S2 = Z1 Z2
# ============================================================

S1 = Z0 @ Z1

S2 = Z1 @ Z2


# ============================================================
# 4. SINGLE-QUBIT STATES
# ============================================================

ket0 = np.array([
    1,
    0
], dtype=complex)

ket1 = np.array([
    0,
    1
], dtype=complex)


# ============================================================
# 5. INITIAL LOGICAL STATE
#
# We use:
#
# |psi> = alpha|0> + beta|1>
#
# Choose a non-trivial superposition.
# ============================================================

alpha = 1 / np.sqrt(3)

beta = np.sqrt(2 / 3)

logical_state = (
    alpha * ket0
    + beta * ket1
)


# ============================================================
# 6. ENCODING
#
# |0> -> |000>
# |1> -> |111>
#
# Therefore:
#
# alpha|0> + beta|1>
#
# becomes:
#
# alpha|000> + beta|111>
# ============================================================

ket000 = np.zeros(
    8,
    dtype=complex
)

ket111 = np.zeros(
    8,
    dtype=complex
)

ket000[0] = 1

ket111[7] = 1


encoded_state = (
    alpha * ket000
    + beta * ket111
)


# ============================================================
# 7. STATE DISPLAY FUNCTION
# ============================================================

def display_state(
    state,
    title
):

    print(title)

    for index, amplitude in enumerate(state):

        if abs(amplitude) > 1e-10:

            binary = format(
                index,
                "03b"
            )

            print(
                f"  |{binary}> : "
                f"{amplitude.real:.6f}"
                f"{amplitude.imag:+.6f}i"
            )

    print()


# ============================================================
# 8. APPLY BIT-FLIP ERROR
# ============================================================

def apply_error(
    state,
    qubit
):

    if qubit == 0:

        return X0 @ state

    elif qubit == 1:

        return X1 @ state

    elif qubit == 2:

        return X2 @ state

    else:

        return state.copy()


# ============================================================
# 9. SYNDROME MEASUREMENT
#
# We calculate the eigenvalue of:
#
# S1 = Z0 Z1
# S2 = Z1 Z2
#
# Expected values:
#
# 00 -> no error
# 10 -> qubit 0
# 11 -> qubit 1
# 01 -> qubit 2
# ============================================================

def calculate_syndrome(
    state
):

    expectation_s1 = np.real(
        np.vdot(
            state,
            S1 @ state
        )
    )

    expectation_s2 = np.real(
        np.vdot(
            state,
            S2 @ state
        )
    )

    # Convert +1 / -1 into syndrome bits.
    #
    # +1 -> 0
    # -1 -> 1

    syndrome_bit_1 = (
        0
        if expectation_s1 > 0
        else 1
    )

    syndrome_bit_2 = (
        0
        if expectation_s2 > 0
        else 1
    )

    syndrome = (
        syndrome_bit_1,
        syndrome_bit_2
    )

    return syndrome


# ============================================================
# 10. IDENTIFY ERROR FROM SYNDROME
# ============================================================

def identify_error(
    syndrome
):

    syndrome_to_qubit = {

        (0, 0): None,

        (1, 0): 0,

        (1, 1): 1,

        (0, 1): 2
    }

    return syndrome_to_qubit.get(
        syndrome,
        None
    )


# ============================================================
# 11. CORRECT ERROR
# ============================================================

def correct_error(
    state,
    qubit
):

    if qubit is None:

        return state.copy()

    if qubit == 0:

        return X0 @ state

    elif qubit == 1:

        return X1 @ state

    elif qubit == 2:

        return X2 @ state

    return state.copy()


# ============================================================
# 12. FIDELITY
#
# For pure states:
#
# Fidelity = |<psi|phi>|^2
# ============================================================

def fidelity(
    state1,
    state2
):

    overlap = np.vdot(
        state1,
        state2
    )

    return abs(overlap) ** 2


# ============================================================
# 13. DECODE
#
# After correction, the valid logical states are:
#
# |000>
# |111>
#
# We calculate the logical probabilities.
# ============================================================

def decode(
    state
):

    probability_000 = abs(
        state[0]
    ) ** 2

    probability_111 = abs(
        state[7]
    ) ** 2

    return (
        probability_000,
        probability_111
    )


# ============================================================
# 14. HEADER
# ============================================================

print("=" * 70)
print("DAY 52 - 3-QUBIT BIT-FLIP QUANTUM ERROR CORRECTION")
print("=" * 70)

print()


# ============================================================
# 15. DISPLAY LOGICAL STATE
# ============================================================

print("LOGICAL QUBIT")
print("-" * 70)

print(
    f"alpha = {alpha:.6f}"
)

print(
    f"beta  = {beta:.6f}"
)

print()

print(
    "|psi> = "
    f"{alpha:.6f}|0> + "
    f"{beta:.6f}|1>"
)

print()


# ============================================================
# 16. DISPLAY ENCODED STATE
# ============================================================

print("ENCODING")
print("-" * 70)

print(
    "|0> -> |000>"
)

print(
    "|1> -> |111>"
)

print()

display_state(
    encoded_state,
    "Encoded logical state:"
)


# ============================================================
# 17. TEST DIFFERENT ERROR LOCATIONS
# ============================================================

error_locations = [
    None,
    0,
    1,
    2
]


for error_qubit in error_locations:

    print("=" * 70)

    if error_qubit is None:

        print("CASE: NO ERROR")

    else:

        print(
            f"CASE: BIT-FLIP ON QUBIT {error_qubit}"
        )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Apply error
    # --------------------------------------------------------

    corrupted_state = apply_error(
        encoded_state,
        error_qubit
    )

    display_state(
        corrupted_state,
        "State after error:"
    )

    # --------------------------------------------------------
    # Calculate syndrome
    # --------------------------------------------------------

    syndrome = calculate_syndrome(
        corrupted_state
    )

    print(
        f"Syndrome = {syndrome}"
    )

    print()

    # --------------------------------------------------------
    # Identify corrupted qubit
    # --------------------------------------------------------

    detected_qubit = identify_error(
        syndrome
    )

    if detected_qubit is None:

        print(
            "Detected error: NONE"
        )

    else:

        print(
            f"Detected error: "
            f"qubit {detected_qubit}"
        )

    print()

    # --------------------------------------------------------
    # Correct
    # --------------------------------------------------------

    corrected_state = correct_error(
        corrupted_state,
        detected_qubit
    )

    display_state(
        corrected_state,
        "State after correction:"
    )

    # --------------------------------------------------------
    # Fidelity
    # --------------------------------------------------------

    recovery_fidelity = fidelity(
        encoded_state,
        corrected_state
    )

    print(
        f"Recovery fidelity = "
        f"{recovery_fidelity:.10f}"
    )

    print()


# ============================================================
# 18. TEST DOUBLE BIT-FLIP
#
# This demonstrates the limitation of the simple
# 3-qubit bit-flip code.
# ============================================================

print("=" * 70)
print("LIMITATION TEST: TWO BIT-FLIP ERRORS")
print("=" * 70)

print()

print(
    "We intentionally apply bit flips to "
    "qubits 0 and 1."
)

print()

double_error_state = X1 @ X0 @ encoded_state


display_state(
    double_error_state,
    "State after two errors:"
)


double_syndrome = calculate_syndrome(
    double_error_state
)

print(
    f"Syndrome = {double_syndrome}"
)

print()

detected_double_error = identify_error(
    double_syndrome
)

if detected_double_error is None:

    print(
        "Decoder thinks there is no error."
    )

else:

    print(
        "Decoder identifies the error as "
        f"qubit {detected_double_error}."
    )

print()

double_corrected_state = correct_error(
    double_error_state,
    detected_double_error
)

display_state(
    double_corrected_state,
    "State after attempted correction:"
)


double_fidelity = fidelity(
    encoded_state,
    double_corrected_state
)

print(
    f"Recovery fidelity = "
    f"{double_fidelity:.10f}"
)

print()


# ============================================================
# 19. SYNDROME TABLE
# ============================================================

print("=" * 70)
print("SYNDROME TABLE")
print("=" * 70)

print()

print(
    f"{'S1':<8}"
    f"{'S2':<8}"
    f"{'Meaning':<25}"
)

print("-" * 45)

print(
    f"{'+1':<8}"
    f"{'+1':<8}"
    f"{'No error':<25}"
)

print(
    f"{'-1':<8}"
    f"{'+1':<8}"
    f"{'Bit flip on q0':<25}"
)

print(
    f"{'-1':<8}"
    f"{'-1':<8}"
    f"{'Bit flip on q1':<25}"
)

print(
    f"{'+1':<8}"
    f"{'-1':<8}"
    f"{'Bit flip on q2':<25}"
)

print()


# ============================================================
# 20. CODE CAPABILITY
# ============================================================

print("=" * 70)
print("CODE CAPABILITY")
print("=" * 70)

print()

print(
    "Corrects:"
)

print(
    "  Single bit-flip error"
)

print()

print(
    "Does NOT reliably correct:"
)

print(
    "  Two simultaneous bit-flip errors"
)

print()

print(
    "This code specifically protects against "
    "bit-flip (X) errors."
)

print()


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("DAY 52 SUMMARY")
print("=" * 70)

print()

print(
    "1. One logical qubit is encoded into "
    "three physical qubits."
)

print(
    "2. |0> is encoded as |000>."
)

print(
    "3. |1> is encoded as |111>."
)

print(
    "4. Parity checks generate an error syndrome."
)

print(
    "5. The syndrome identifies the location "
    "of one bit-flip error."
)

print(
    "6. An X operation corrects the detected error."
)

print(
    "7. The code cannot reliably correct "
    "two simultaneous bit flips."
)

print()

print(
    "Error Correction:"
)

print(
    "Encoding -> Error -> Syndrome -> Correction"
)

print()

print("=" * 70)
print("DAY 52 COMPLETE")
print("=" * 70)