import numpy as np


# ============================================================
# DAY 51 - QUANTUM NOISE CHANNELS
#
# We study:
#   1. Bit-flip channel  (X error)
#   2. Phase-flip channel (Z error)
#   3. Bit-phase-flip channel (Y error)
#
# We analyze their effect using:
#   - Density matrices
#   - Pauli expectation values
#   - Bloch-vector components
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
# 2. INITIAL STATE
#
# We use:
#
# |+> = (|0> + |1>) / sqrt(2)
#
# This state has:
#
# <X> = 1
# <Y> = 0
# <Z> = 0
# ============================================================

plus_state = np.array([
    1,
    1
], dtype=complex) / np.sqrt(2)


# Density matrix:
#
# rho = |psi><psi|

rho_ideal = np.outer(
    plus_state,
    plus_state.conj()
)


# ============================================================
# 3. EXPECTATION VALUE
# ============================================================

def expectation(
    rho,
    operator
):

    return np.real(
        np.trace(rho @ operator)
    )


# ============================================================
# 4. BIT-FLIP CHANNEL
#
# rho' = (1-p)rho + p X rho X
#
# With probability:
#
# 1-p -> no error
# p   -> X error
# ============================================================

def bit_flip_channel(
    rho,
    p
):

    return (
        (1 - p) * rho
        + p * X @ rho @ X
    )


# ============================================================
# 5. PHASE-FLIP CHANNEL
#
# rho' = (1-p)rho + p Z rho Z
#
# With probability:
#
# 1-p -> no error
# p   -> Z error
# ============================================================

def phase_flip_channel(
    rho,
    p
):

    return (
        (1 - p) * rho
        + p * Z @ rho @ Z
    )


# ============================================================
# 6. BIT-PHASE-FLIP CHANNEL
#
# rho' = (1-p)rho + p Y rho Y
#
# Y combines bit-flip and phase-flip behavior.
# ============================================================

def bit_phase_flip_channel(
    rho,
    p
):

    return (
        (1 - p) * rho
        + p * Y @ rho @ Y
    )


# ============================================================
# 7. DEPOLARIZING CHANNEL
#
# Included for comparison with the noise model used
# in our previous VQE/error-mitigation experiments.
#
# rho' = (1-p)rho + p I/2
# ============================================================

def depolarizing_channel(
    rho,
    p
):

    return (
        (1 - p) * rho
        + p * I / 2
    )


# ============================================================
# 8. DISPLAY EXPECTATIONS
# ============================================================

def display_expectations(
    name,
    rho
):

    exp_x = expectation(
        rho,
        X
    )

    exp_y = expectation(
        rho,
        Y
    )

    exp_z = expectation(
        rho,
        Z
    )

    print(name)

    print(
        f"  <X> = {exp_x:.6f}"
    )

    print(
        f"  <Y> = {exp_y:.6f}"
    )

    print(
        f"  <Z> = {exp_z:.6f}"
    )

    print()


# ============================================================
# 9. DISPLAY DENSITY MATRIX
# ============================================================

def display_density_matrix(
    name,
    rho
):

    print(name)

    print(
        np.round(
            rho,
            6
        )
    )

    print()


# ============================================================
# 10. NOISE PARAMETERS
# ============================================================

noise_strengths = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0
]


# ============================================================
# 11. HEADER
# ============================================================

print("=" * 70)
print("DAY 51 - QUANTUM NOISE CHANNELS")
print("=" * 70)

print()

print(
    "Initial state: |+> = "
    "(|0> + |1>) / sqrt(2)"
)

print()

display_density_matrix(
    "Ideal density matrix:",
    rho_ideal
)

display_expectations(
    "Ideal expectation values:",
    rho_ideal
)


# ============================================================
# 12. SINGLE EXAMPLE
#
# Use p = 0.5 to clearly see the effect.
# ============================================================

p_example = 0.5

print("=" * 70)
print(
    f"EXAMPLE AT p = {p_example}"
)
print("=" * 70)

print()

rho_bit = bit_flip_channel(
    rho_ideal,
    p_example
)

rho_phase = phase_flip_channel(
    rho_ideal,
    p_example
)

rho_bit_phase = bit_phase_flip_channel(
    rho_ideal,
    p_example
)

rho_depolarizing = depolarizing_channel(
    rho_ideal,
    p_example
)


display_expectations(
    "Bit-flip channel:",
    rho_bit
)

display_expectations(
    "Phase-flip channel:",
    rho_phase
)

display_expectations(
    "Bit-phase-flip channel:",
    rho_bit_phase
)

display_expectations(
    "Depolarizing channel:",
    rho_depolarizing
)


# ============================================================
# 13. NOISE SWEEP
#
# Study how the X, Y and Z expectation values change
# as the noise probability increases.
# ============================================================

print("=" * 70)
print("NOISE SWEEP")
print("=" * 70)

print()


for p in noise_strengths:

    print("-" * 70)

    print(
        f"Noise probability p = {p:.2f}"
    )

    print()

    # ----------------------------------------
    # Bit flip
    # ----------------------------------------

    rho_bit = bit_flip_channel(
        rho_ideal,
        p
    )

    bit_x = expectation(
        rho_bit,
        X
    )

    bit_y = expectation(
        rho_bit,
        Y
    )

    bit_z = expectation(
        rho_bit,
        Z
    )

    # ----------------------------------------
    # Phase flip
    # ----------------------------------------

    rho_phase = phase_flip_channel(
        rho_ideal,
        p
    )

    phase_x = expectation(
        rho_phase,
        X
    )

    phase_y = expectation(
        rho_phase,
        Y
    )

    phase_z = expectation(
        rho_phase,
        Z
    )

    # ----------------------------------------
    # Bit-phase flip
    # ----------------------------------------

    rho_bit_phase = bit_phase_flip_channel(
        rho_ideal,
        p
    )

    bit_phase_x = expectation(
        rho_bit_phase,
        X
    )

    bit_phase_y = expectation(
        rho_bit_phase,
        Y
    )

    bit_phase_z = expectation(
        rho_bit_phase,
        Z
    )

    print(
        "Bit-flip:"
    )

    print(
        f"  <X> = {bit_x:.6f}, "
        f"<Y> = {bit_y:.6f}, "
        f"<Z> = {bit_z:.6f}"
    )

    print()

    print(
        "Phase-flip:"
    )

    print(
        f"  <X> = {phase_x:.6f}, "
        f"<Y> = {phase_y:.6f}, "
        f"<Z> = {phase_z:.6f}"
    )

    print()

    print(
        "Bit-phase-flip:"
    )

    print(
        f"  <X> = {bit_phase_x:.6f}, "
        f"<Y> = {bit_phase_y:.6f}, "
        f"<Z> = {bit_phase_z:.6f}"
    )

    print()


# ============================================================
# 14. ANALYTICAL CHECK
#
# For |+>:
#
# Bit flip:
#   <X> remains 1
#
# Phase flip:
#   <X> = 1 - 2p
#
# Bit-phase flip:
#   <X> = 1 - 2p
# ============================================================

print("=" * 70)
print("ANALYTICAL CHECK")
print("=" * 70)

print()

print(
    "For the |+> state:"
)

print()

print(
    "Bit-flip channel:"
)

print(
    "  <X> = 1"
)

print(
    "  <Y> = 0"
)

print(
    "  <Z> = 0"
)

print()

print(
    "Phase-flip channel:"
)

print(
    "  <X> = 1 - 2p"
)

print(
    "  <Y> = 0"
)

print(
    "  <Z> = 0"
)

print()

print(
    "Bit-phase-flip channel:"
)

print(
    "  <X> = 1 - 2p"
)

print(
    "  <Y> = 0"
)

print(
    "  <Z> = 0"
)

print()


# ============================================================
# 15. BLOCH VECTOR INTERPRETATION
# ============================================================

print("=" * 70)
print("BLOCH VECTOR INTERPRETATION")
print("=" * 70)

print()

print(
    "The Bloch vector is:"
)

print(
    "    r = (<X>, <Y>, <Z>)"
)

print()

print(
    "For the ideal |+> state:"
)

print(
    "    r = (1, 0, 0)"
)

print()

print(
    "Therefore |+> lies along the +X direction "
    "of the Bloch sphere."
)

print()

print(
    "Different noise channels affect different "
    "components of this vector."
)

print()


# ============================================================
# 16. COMPARISON TABLE
# ============================================================

print("=" * 70)
print("NOISE CHANNEL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Channel':<25}"
    f"{'Operation':<20}"
    f"{'Main effect'}"
)

print("-" * 70)

print(
    f"{'Bit flip':<25}"
    f"{'X':<20}"
    f"{'|0> <-> |1>'}"
)

print(
    f"{'Phase flip':<25}"
    f"{'Z':<20}"
    f"{'Relative phase change'}"
)

print(
    f"{'Bit-phase flip':<25}"
    f"{'Y':<20}"
    f"{'Bit + phase change'}"
)

print(
    f"{'Depolarizing':<25}"
    f"{'Mixed noise':<20}"
    f"{'State becomes mixed'}"
)

print()


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("DAY 51 SUMMARY")
print("=" * 70)

print()

print(
    "1. Bit-flip noise applies an X error."
)

print(
    "2. Phase-flip noise applies a Z error."
)

print(
    "3. Bit-phase-flip noise applies a Y error."
)

print(
    "4. Noise channels can be represented using "
    "density matrices."
)

print(
    "5. Pauli expectation values show how the "
    "quantum state changes."
)

print(
    "6. The Bloch vector provides a geometric "
    "interpretation of the noise."
)

print()

print("=" * 70)
print("DAY 51 COMPLETE")
print("=" * 70)