import numpy as np

# ============================================================
# DAY 53 — PHASE-FLIP QUANTUM ERROR CORRECTION
# 3-Qubit Phase-Flip Code
# ============================================================

np.set_printoptions(precision=6, suppress=True)


# ------------------------------------------------------------
# 1. BASIC MATRICES
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. THREE-QUBIT OPERATORS
# ------------------------------------------------------------

X0 = np.kron(np.kron(X, I), I)
X1 = np.kron(np.kron(I, X), I)
X2 = np.kron(np.kron(I, I), X)

Z0 = np.kron(np.kron(Z, I), I)
Z1 = np.kron(np.kron(I, Z), I)
Z2 = np.kron(np.kron(I, I), Z)

H0 = np.kron(np.kron(H, I), I)
H1 = np.kron(np.kron(I, H), I)
H2 = np.kron(np.kron(I, I), H)

# Phase-flip code syndrome operators
S1 = X0 @ X1
S2 = X1 @ X2


# ------------------------------------------------------------
# 3. LOGICAL QUBIT
# ------------------------------------------------------------

alpha = 1 / np.sqrt(3)
beta = np.sqrt(2 / 3)

logical_state = np.array([
    alpha,
    beta
], dtype=complex)

print("=" * 65)
print("DAY 53 — PHASE-FLIP QUANTUM ERROR CORRECTION")
print("=" * 65)

print("\nLogical state:")
print("|psi> = alpha|0> + beta|1>")

print(f"alpha = {alpha:.6f}")
print(f"beta  = {beta:.6f}")

print(f"\nNormalization = {np.linalg.norm(logical_state)**2:.6f}")


# ------------------------------------------------------------
# 4. X-BASIS STATES
# ------------------------------------------------------------

plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
minus = np.array([1, -1], dtype=complex) / np.sqrt(2)


# ------------------------------------------------------------
# 5. THREE-QUBIT X-BASIS STATES
# ------------------------------------------------------------

plus_plus_plus = np.kron(
    np.kron(plus, plus),
    plus
)

minus_minus_minus = np.kron(
    np.kron(minus, minus),
    minus
)


# ------------------------------------------------------------
# 6. ENCODE THE LOGICAL QUBIT
# ------------------------------------------------------------

encoded_state = (
    alpha * plus_plus_plus
    + beta * minus_minus_minus
)

print("\nEncoded state:")
print("|psi_L> = alpha|+++> + beta|--->")

print("\nEncoded state vector:")
print(encoded_state)

print(
    f"\nEncoded state normalization = "
    f"{np.linalg.norm(encoded_state)**2:.6f}"
)


# ------------------------------------------------------------
# 7. SYNDROME CALCULATION
# ------------------------------------------------------------

def get_syndrome(state):
    """
    Calculate the eigenvalue of each syndrome operator.

    S1 = X0 X1
    S2 = X1 X2

    Eigenvalues:
        +1 -> syndrome bit 0
        -1 -> syndrome bit 1
    """

    s1_value = np.real(np.vdot(state, S1 @ state))
    s2_value = np.real(np.vdot(state, S2 @ state))

    # Round to avoid floating-point noise
    s1_value = 1 if s1_value >= 0 else -1
    s2_value = 1 if s2_value >= 0 else -1

    syndrome = (
        0 if s1_value == 1 else 1,
        0 if s2_value == 1 else 1
    )

    return syndrome


# ------------------------------------------------------------
# 8. DETERMINE WHICH QUBIT HAS THE ERROR
# ------------------------------------------------------------

def identify_error(syndrome):

    syndrome_map = {
        (0, 0): "No error",
        (1, 0): "Qubit 0",
        (1, 1): "Qubit 1",
        (0, 1): "Qubit 2"
    }

    return syndrome_map[syndrome]


# ------------------------------------------------------------
# 9. APPLY CORRECTION
# ------------------------------------------------------------

def correct_error(state, syndrome):

    if syndrome == (1, 0):
        print("Applying Z correction to qubit 0...")
        return Z0 @ state

    elif syndrome == (1, 1):
        print("Applying Z correction to qubit 1...")
        return Z1 @ state

    elif syndrome == (0, 1):
        print("Applying Z correction to qubit 2...")
        return Z2 @ state

    else:
        print("No correction required.")
        return state


# ------------------------------------------------------------
# 10. FIDELITY
# ------------------------------------------------------------

def fidelity(state1, state2):

    overlap = np.vdot(state1, state2)

    return abs(overlap) ** 2


# ------------------------------------------------------------
# 11. TEST DIFFERENT ERROR CASES
# ------------------------------------------------------------

def test_error(error_name, error_operator):

    print("\n" + "-" * 65)
    print(f"TEST: {error_name}")
    print("-" * 65)

    # Apply error
    corrupted_state = error_operator @ encoded_state

    # Find syndrome
    syndrome = get_syndrome(corrupted_state)

    print(f"Syndrome = {syndrome}")

    # Identify error
    detected_error = identify_error(syndrome)

    print(f"Detected = {detected_error}")

    # Correct
    corrected_state = correct_error(
        corrupted_state,
        syndrome
    )

    # Calculate fidelity
    F = fidelity(
        encoded_state,
        corrected_state
    )

    print(f"Fidelity = {F:.6f}")

    if abs(F - 1.0) < 1e-10:
        print("SUCCESS: Error completely corrected.")
    else:
        print("ERROR: Correction was not perfect.")

    return corrected_state


# ------------------------------------------------------------
# 12. NO ERROR
# ------------------------------------------------------------

test_error(
    "No error",
    np.eye(8, dtype=complex)
)


# ------------------------------------------------------------
# 13. PHASE FLIP ON QUBIT 0
# ------------------------------------------------------------

test_error(
    "Phase flip Z on qubit 0",
    Z0
)


# ------------------------------------------------------------
# 14. PHASE FLIP ON QUBIT 1
# ------------------------------------------------------------

test_error(
    "Phase flip Z on qubit 1",
    Z1
)


# ------------------------------------------------------------
# 15. PHASE FLIP ON QUBIT 2
# ------------------------------------------------------------

test_error(
    "Phase flip Z on qubit 2",
    Z2
)


# ------------------------------------------------------------
# 16. DOUBLE PHASE FLIP
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("DOUBLE PHASE-FLIP TEST")
print("=" * 65)

double_error_operator = Z0 @ Z1

double_corrupted_state = (
    double_error_operator @ encoded_state
)

double_syndrome = get_syndrome(
    double_corrupted_state
)

print(f"\nErrors applied: Z0 + Z1")
print(f"Syndrome = {double_syndrome}")

print(
    f"Decoder interpretation = "
    f"{identify_error(double_syndrome)}"
)

double_corrected_state = correct_error(
    double_corrupted_state,
    double_syndrome
)

double_fidelity = fidelity(
    encoded_state,
    double_corrected_state
)

print(f"Fidelity after attempted correction = {double_fidelity:.6f}")

print("\nImportant:")
print("The 3-qubit phase-flip code can correct")
print("ONE phase-flip error, but not TWO phase-flip errors.")


# ------------------------------------------------------------
# 17. SYNDROME TABLE
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("PHASE-FLIP SYNDROME TABLE")
print("=" * 65)

print("\nS1 = X0 X1")
print("S2 = X1 X2")

print("\nSyndrome     Meaning")
print("----------------------------")
print("(0, 0)       No error")
print("(1, 0)       Z error on Qubit 0")
print("(1, 1)       Z error on Qubit 1")
print("(0, 1)       Z error on Qubit 2")


# ------------------------------------------------------------
# 18. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("DAY 53 SUMMARY")
print("=" * 65)

print("""
Phase-flip code:

Logical encoding:
    |0> -> |+++>
    |1> -> |--->

Encoded state:
    |psi_L> = alpha|+++> + beta|--->

Syndrome operators:
    S1 = X0 X1
    S2 = X1 X2

Correction operators:
    Z0, Z1, Z2

Capability:
    Corrects ONE phase-flip error.
    Cannot reliably correct TWO phase-flip errors.

Key idea:
    A phase flip in the computational basis
    behaves like a bit flip in the X basis.
""")

print("=" * 65)