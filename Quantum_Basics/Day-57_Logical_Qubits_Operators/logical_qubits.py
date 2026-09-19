import numpy as np

# ============================================================
# DAY 57 — LOGICAL QUBITS & LOGICAL OPERATORS
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


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def kron_all(operators):
    result = operators[0]

    for operator in operators[1:]:
        result = np.kron(result, operator)

    return result


def expectation(state, operator):
    return np.vdot(state, operator @ state)


def fidelity(state1, state2):
    return abs(np.vdot(state1, state2)) ** 2


def same_state_up_to_global_phase(
    state1,
    state2,
    tolerance=1e-10
):
    overlap = np.vdot(state1, state2)

    if abs(overlap) < tolerance:
        return False

    phase = overlap / abs(overlap)

    return np.allclose(
        state1,
        phase * state2,
        atol=tolerance
    )


# ============================================================
# 3. SINGLE-QUBIT BASIS STATES
# ============================================================

zero = np.array([1, 0], dtype=complex)

one = np.array([0, 1], dtype=complex)


# ============================================================
# 4. THREE-QUBIT BASIS STATES
# ============================================================

zero_zero_zero = kron_all([
    zero,
    zero,
    zero
])

one_one_one = kron_all([
    one,
    one,
    one
])


# ============================================================
# 5. LOGICAL BASIS STATES
#
# 3-qubit bit-flip code:
#
# |0_L> = |000>
# |1_L> = |111>
# ============================================================

logical_zero = zero_zero_zero
logical_one = one_one_one


print("=" * 70)
print("DAY 57 — LOGICAL QUBITS & LOGICAL OPERATORS")
print("=" * 70)


# ============================================================
# 6. DISPLAY LOGICAL STATES
# ============================================================

print("\n" + "-" * 70)
print("1. LOGICAL BASIS STATES")
print("-" * 70)

print("""
3-qubit bit-flip code:

    |0_L> = |000>
    |1_L> = |111>
""")

print("\n|0_L> =")
print(logical_zero)

print("\n|1_L> =")
print(logical_one)

print(
    f"\n||0_L||² = "
    f"{np.linalg.norm(logical_zero) ** 2:.6f}"
)

print(
    f"||1_L||² = "
    f"{np.linalg.norm(logical_one) ** 2:.6f}"
)


# ============================================================
# 7. ARBITRARY LOGICAL QUBIT
# ============================================================

alpha = 1 / np.sqrt(3)
beta = np.sqrt(2 / 3)

logical_state = (
    alpha * logical_zero
    + beta * logical_one
)

print("\n" + "-" * 70)
print("2. ARBITRARY LOGICAL QUBIT")
print("-" * 70)

print("""
|psi_L> = alpha|0_L> + beta|1_L>
""")

print(f"alpha = {alpha:.6f}")
print(f"beta  = {beta:.6f}")

print("\nLogical state:")
print(logical_state)

print(
    f"\nNormalization = "
    f"{np.linalg.norm(logical_state) ** 2:.6f}"
)


# ============================================================
# 8. PHYSICAL QUBIT OPERATORS
# ============================================================

X0 = kron_all([X, I, I])
X1 = kron_all([I, X, I])
X2 = kron_all([I, I, X])

Y0 = kron_all([Y, I, I])
Y1 = kron_all([I, Y, I])
Y2 = kron_all([I, I, Y])

Z0 = kron_all([Z, I, I])
Z1 = kron_all([I, Z, I])
Z2 = kron_all([I, I, Z])


# ============================================================
# 9. STABILIZERS
#
# S1 = Z0 Z1
# S2 = Z1 Z2
# ============================================================

S1 = Z0 @ Z1
S2 = Z1 @ Z2

print("\n" + "-" * 70)
print("3. STABILIZERS")
print("-" * 70)

print("""
S1 = Z0 Z1
S2 = Z1 Z2
""")


# ============================================================
# 10. VERIFY STABILIZERS
# ============================================================

print("Checking stabilizers on |0_L>:")

if same_state_up_to_global_phase(
    S1 @ logical_zero,
    logical_zero
):
    print("S1|0_L> = |0_L>")

if same_state_up_to_global_phase(
    S2 @ logical_zero,
    logical_zero
):
    print("S2|0_L> = |0_L>")


print("\nChecking stabilizers on |1_L>:")

if same_state_up_to_global_phase(
    S1 @ logical_one,
    logical_one
):
    print("S1|1_L> = |1_L>")

if same_state_up_to_global_phase(
    S2 @ logical_one,
    logical_one
):
    print("S2|1_L> = |1_L>")


print("\nStabilizers on arbitrary logical state:")

print(
    f"<S1> = "
    f"{np.real(expectation(logical_state, S1)):.6f}"
)

print(
    f"<S2> = "
    f"{np.real(expectation(logical_state, S2)):.6f}"
)


# ============================================================
# 11. LOGICAL X
#
# X_L = X0 X1 X2
# ============================================================

X_L = X0 @ X1 @ X2

print("\n" + "-" * 70)
print("4. LOGICAL X")
print("-" * 70)

print("""
X_L = X0 X1 X2
""")


logical_x_zero = X_L @ logical_zero
logical_x_one = X_L @ logical_one


print("X_L|0_L> =")
print(logical_x_zero)

print("\nExpected |1_L> =")
print(logical_one)

if same_state_up_to_global_phase(
    logical_x_zero,
    logical_one
):
    print("\nSUCCESS: X_L|0_L> = |1_L>")


print("\nX_L|1_L> =")
print(logical_x_one)

print("\nExpected |0_L> =")
print(logical_zero)

if same_state_up_to_global_phase(
    logical_x_one,
    logical_zero
):
    print("\nSUCCESS: X_L|1_L> = |0_L>")


# ============================================================
# 12. LOGICAL Z
#
# Z_L = Z0
# ============================================================

Z_L = Z0

print("\n" + "-" * 70)
print("5. LOGICAL Z")
print("-" * 70)

print("""
Z_L = Z0
""")


logical_z_zero = Z_L @ logical_zero
logical_z_one = Z_L @ logical_one


print("Z_L|0_L> =")
print(logical_z_zero)

print("\nExpected |0_L> =")
print(logical_zero)

if same_state_up_to_global_phase(
    logical_z_zero,
    logical_zero
):
    print("\nSUCCESS: Z_L|0_L> = |0_L>")


print("\nZ_L|1_L> =")
print(logical_z_one)

print("\nExpected -|1_L> =")
print(-logical_one)

if np.allclose(
    logical_z_one,
    -logical_one
):
    print("\nSUCCESS: Z_L|1_L> = -|1_L>")


# ============================================================
# 13. LOGICAL X ON ARBITRARY STATE
# ============================================================

print("\n" + "-" * 70)
print("6. LOGICAL X ON ARBITRARY STATE")
print("-" * 70)

after_logical_x = X_L @ logical_state

expected_logical_x = (
    alpha * logical_one
    + beta * logical_zero
)

if same_state_up_to_global_phase(
    after_logical_x,
    expected_logical_x
):
    print(
        "SUCCESS: Logical X acts correctly."
    )

print("\nX_L|psi_L> produces:")
print(
    "alpha|1_L> + beta|0_L>"
)


# ============================================================
# 14. LOGICAL Z ON ARBITRARY STATE
# ============================================================

print("\n" + "-" * 70)
print("7. LOGICAL Z ON ARBITRARY STATE")
print("-" * 70)

after_logical_z = Z_L @ logical_state

expected_logical_z = (
    alpha * logical_zero
    - beta * logical_one
)

if same_state_up_to_global_phase(
    after_logical_z,
    expected_logical_z
):
    print(
        "SUCCESS: Logical Z acts correctly."
    )

print("\nZ_L|psi_L> produces:")
print(
    "alpha|0_L> - beta|1_L>"
)


# ============================================================
# 15. LOGICAL Y
#
# Y_L = i X_L Z_L
# ============================================================

Y_L = 1j * X_L @ Z_L

print("\n" + "-" * 70)
print("8. LOGICAL Y")
print("-" * 70)

print("""
Y_L = i X_L Z_L
""")


logical_y_zero = Y_L @ logical_zero
logical_y_one = Y_L @ logical_one

print("\nY_L|0_L> =")
print(logical_y_zero)

print("\nY_L|1_L> =")
print(logical_y_one)


# ============================================================
# 16. LOGICAL PAULI ALGEBRA
# ============================================================

print("\n" + "-" * 70)
print("9. LOGICAL PAULI ALGEBRA")
print("-" * 70)

XL_squared = X_L @ X_L
ZL_squared = Z_L @ Z_L
YL_squared = Y_L @ Y_L

identity_3 = np.eye(8, dtype=complex)


if np.allclose(
    XL_squared,
    identity_3
):
    print("SUCCESS: X_L² = I")

if np.allclose(
    ZL_squared,
    identity_3
):
    print("SUCCESS: Z_L² = I")

if np.allclose(
    YL_squared,
    identity_3
):
    print("SUCCESS: Y_L² = I")


# ============================================================
# 17. LOGICAL X AND Z ANTICOMMUTATION
# ============================================================

print("\n" + "-" * 70)
print("10. LOGICAL X AND Z ANTICOMMUTATION")
print("-" * 70)

XLZL = X_L @ Z_L
ZLXL = Z_L @ X_L

if np.allclose(
    XLZL,
    -ZLXL
):
    print(
        "SUCCESS: X_L and Z_L ANTICOMMUTE."
    )

    print(
        "X_L Z_L = -Z_L X_L"
    )


# ============================================================
# 18. MULTIPLE REPRESENTATIONS OF LOGICAL Z
# ============================================================

print("\n" + "-" * 70)
print("11. MULTIPLE REPRESENTATIONS OF LOGICAL Z")
print("-" * 70)

print("""
Inside the code space, Z0, Z1 and Z2
all perform the same logical Z operation.
""")


for name, operator in [
    ("Z0", Z0),
    ("Z1", Z1),
    ("Z2", Z2)
]:

    result_zero = operator @ logical_zero
    result_one = operator @ logical_one

    correct_zero = same_state_up_to_global_phase(
        result_zero,
        logical_zero
    )

    correct_one = np.allclose(
        result_one,
        -logical_one
    )

    print(
        f"{name}: "
        f"|0_L> -> |0_L> = {correct_zero}, "
        f"|1_L> -> -|1_L> = {correct_one}"
    )


# ============================================================
# 19. STABILIZER PRODUCT
# ============================================================

print("\n" + "-" * 70)
print("12. STABILIZER PRODUCT")
print("-" * 70)

print("""
S1 = Z0 Z1
S2 = Z1 Z2

Therefore:

S1 S2
= (Z0 Z1)(Z1 Z2)
= Z0 Z2

because:

Z1² = I
""")


S1S2 = S1 @ S2
Z0Z2 = Z0 @ Z2

if np.allclose(
    S1S2,
    Z0Z2
):
    print("SUCCESS: S1 S2 = Z0 Z2")


# ============================================================
# 20. FIDELITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("13. FIDELITY CHECKS")
print("=" * 70)

F_X_zero = fidelity(
    X_L @ logical_zero,
    logical_one
)

F_X_one = fidelity(
    X_L @ logical_one,
    logical_zero
)

F_Z_zero = fidelity(
    Z_L @ logical_zero,
    logical_zero
)

F_Z_one = fidelity(
    Z_L @ logical_one,
    -logical_one
)


print(
    f"\nFidelity X_L|0_L> vs |1_L> = "
    f"{F_X_zero:.6f}"
)

print(
    f"Fidelity X_L|1_L> vs |0_L> = "
    f"{F_X_one:.6f}"
)

print(
    f"Fidelity Z_L|0_L> vs |0_L> = "
    f"{F_Z_zero:.6f}"
)

print(
    f"Fidelity Z_L|1_L> vs -|1_L> = "
    f"{F_Z_one:.6f}"
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 57 SUMMARY")
print("=" * 70)

print("""
LOGICAL QUBIT:

    |0_L> = |000>
    |1_L> = |111>

ARBITRARY LOGICAL STATE:

    |psi_L>
      =
    alpha|0_L> + beta|1_L>


STABILIZERS:

    S1 = Z0 Z1
    S2 = Z1 Z2

They preserve the encoded information.


LOGICAL X:

    X_L = X0 X1 X2

    |0_L> -> |1_L>
    |1_L> -> |0_L>


LOGICAL Z:

    Z_L = Z0

    |0_L> -> |0_L>
    |1_L> -> -|1_L>


LOGICAL Y:

    Y_L = i X_L Z_L


KEY DISTINCTION:

    Stabilizer
        ↓
    Preserves logical information

    Logical operator
        ↓
    Acts on logical information


IMPORTANT:

    Physical qubits = hardware

    Logical qubit = protected information

    Multiple physical qubits can collectively
    represent one logical qubit.
""")

print("=" * 70)
print("DAY 57 COMPLETE")
print("=" * 70)