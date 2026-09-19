import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


# ============================================================
# Two-Qubit VQE - Entangling Ansatz
#
# Hamiltonian:
#
# H = 0.5 Z0 + 0.5 Z1 + 0.8 Z0Z1 + 0.2 X0X1
#
# Exact ground-state energy = -1.0
#
# The previous ansatz could not reach the true ground state.
# This version introduces entanglement explicitly.
# ============================================================


# ============================================================
# Hamiltonian coefficients
# ============================================================

Z0_COEFF = 0.5
Z1_COEFF = 0.5
ZZ_COEFF = 0.8
XX_COEFF = 0.2

SHOTS = 4096


# ============================================================
# Exact Hamiltonian
# ============================================================

HAMILTONIAN = SparsePauliOp.from_list([
    ("ZI", Z0_COEFF),
    ("IZ", Z1_COEFF),
    ("ZZ", ZZ_COEFF),
    ("XX", XX_COEFF),
])


# ============================================================
# Exact ground-state energy
# ============================================================

def calculate_exact_ground_energy():

    matrix = HAMILTONIAN.to_matrix()

    eigenvalues = np.linalg.eigvalsh(matrix)

    return np.min(eigenvalues)


EXACT_GROUND_ENERGY = calculate_exact_ground_energy()


# ============================================================
# Entangling Ansatz
# ============================================================

def build_ansatz(params):

    theta0, theta1, entangle, phi = params

    qc = QuantumCircuit(2)

    # --------------------------------------------------------
    # Single-qubit rotations
    # --------------------------------------------------------

    qc.ry(theta0, 0)
    qc.ry(theta1, 1)

    # --------------------------------------------------------
    # Create entanglement
    # --------------------------------------------------------

    qc.cx(0, 1)

    # Additional variational rotations
    qc.ry(entangle, 0)
    qc.rz(phi, 1)

    # Second entangling operation
    qc.cx(0, 1)

    return qc


# ============================================================
# Statevector energy
#
# Used during optimization.
#
# This removes shot noise from the optimizer so we can
# determine whether the ansatz itself can reach the
# ground state.
# ============================================================

def exact_energy(params):

    qc = build_ansatz(params)

    state = Statevector.from_instruction(qc)

    energy = np.real(
        state.expectation_value(HAMILTONIAN)
    )

    return float(energy)


# ============================================================
# Measurement helper
# ============================================================

def expectation_from_counts(counts, shots):

    expectation = 0.0

    for bitstring, count in counts.items():

        # Qiskit displays:
        #
        # q1 q0
        #
        # so the rightmost bit is q0
        # and the leftmost bit is q1.

        q1 = int(bitstring[0])
        q0 = int(bitstring[1])

        eigenvalue = 1 if q0 == q1 else -1

        expectation += eigenvalue * count

    return expectation / shots


# ============================================================
# Measure Z terms
# ============================================================

def measure_z_terms(params, shots=SHOTS):

    qc = build_ansatz(params)

    qc.measure_all()

    simulator = AerSimulator()

    result = simulator.run(
        qc,
        shots=shots
    ).result()

    counts = result.get_counts()

    z0_total = 0
    z1_total = 0
    zz_total = 0

    for bitstring, count in counts.items():

        q1 = int(bitstring[0])
        q0 = int(bitstring[1])

        z0 = 1 if q0 == 0 else -1
        z1 = 1 if q1 == 0 else -1

        zz = z0 * z1

        z0_total += z0 * count
        z1_total += z1 * count
        zz_total += zz * count

    z0 = z0_total / shots
    z1 = z1_total / shots
    zz = zz_total / shots

    return z0, z1, zz, counts


# ============================================================
# Measure X0 X1
# ============================================================

def measure_xx(params, shots=SHOTS):

    qc = build_ansatz(params)

    # --------------------------------------------------------
    # Rotate X basis -> Z basis
    # --------------------------------------------------------

    qc.h(0)
    qc.h(1)

    qc.measure_all()

    simulator = AerSimulator()

    result = simulator.run(
        qc,
        shots=shots
    ).result()

    counts = result.get_counts()

    xx = expectation_from_counts(
        counts,
        shots
    )

    return xx, counts


# ============================================================
# Full finite-shot energy
# ============================================================

def measured_energy(params, shots=SHOTS):

    z0, z1, zz, z_counts = measure_z_terms(
        params,
        shots
    )

    xx, xx_counts = measure_xx(
        params,
        shots
    )

    energy = (
        Z0_COEFF * z0
        + Z1_COEFF * z1
        + ZZ_COEFF * zz
        + XX_COEFF * xx
    )

    return (
        energy,
        z0,
        z1,
        zz,
        xx,
        z_counts,
        xx_counts
    )


# ============================================================
# Objective function
#
# IMPORTANT:
# We use exact statevector energy here.
# Therefore COBYLA sees a smooth deterministic objective.
# ============================================================

optimization_history = []


def objective(params):

    energy = exact_energy(params)

    optimization_history.append(
        (params.copy(), energy)
    )

    print(
        f"theta0={params[0]:.4f}, "
        f"theta1={params[1]:.4f}, "
        f"entangle={params[2]:.4f}, "
        f"phi={params[3]:.4f} | "
        f"Exact E={energy:.8f}"
    )

    return energy


# ============================================================
# Initial parameters
# ============================================================

initial_parameters = np.array([
    0.5,   # theta0
    0.5,   # theta1
    0.5,   # entangle
    0.0    # phi
])


# ============================================================
# Header
# ============================================================

print()
print("Two-Qubit VQE - Entangling Ansatz")
print("=================================")

print()
print("Hamiltonian:")
print(
    "H = 0.5 Z0 + 0.5 Z1 + "
    "0.8 Z0Z1 + 0.2 X0X1"
)

print()
print(
    f"Exact ground-state energy = "
    f"{EXACT_GROUND_ENERGY:.8f}"
)

print()
print("Initial Parameters:")

print(
    f"theta0   = {initial_parameters[0]:.6f}"
)

print(
    f"theta1   = {initial_parameters[1]:.6f}"
)

print(
    f"entangle = {initial_parameters[2]:.6f}"
)

print(
    f"phi      = {initial_parameters[3]:.6f}"
)


# ============================================================
# Initial exact energy
# ============================================================

initial_exact_energy = exact_energy(
    initial_parameters
)

print()
print(
    f"Initial exact energy = "
    f"{initial_exact_energy:.8f}"
)


# ============================================================
# Initial finite-shot verification
# ============================================================

(
    initial_measured_energy,
    initial_z0,
    initial_z1,
    initial_zz,
    initial_xx,
    initial_z_counts,
    initial_xx_counts
) = measured_energy(
    initial_parameters,
    shots=SHOTS
)


print()
print("Initial Measurements:")
print(
    f"<Z0> = {initial_z0:.6f}"
)

print(
    f"<Z1> = {initial_z1:.6f}"
)

print(
    f"<Z0Z1> = {initial_zz:.6f}"
)

print(
    f"<X0X1> = {initial_xx:.6f}"
)

print(
    f"Measured initial energy = "
    f"{initial_measured_energy:.8f}"
)


# ============================================================
# Optimization
# ============================================================

print()
print("COBYLA Optimization:")
print("====================")


result = minimize(
    objective,
    initial_parameters,
    method="COBYLA",
    options={
        "maxiter": 100,
        "rhobeg": 0.5,
        "tol": 1e-7
    }
)


# ============================================================
# Optimization result
# ============================================================

best_parameters = result.x

best_theta0 = best_parameters[0]
best_theta1 = best_parameters[1]
best_entangle = best_parameters[2]
best_phi = best_parameters[3]

best_ideal_energy = exact_energy(
    best_parameters
)


print()
print("Optimization Complete:")
print("======================")

print(
    f"Best theta0   = "
    f"{best_theta0:.8f}"
)

print(
    f"Best theta1   = "
    f"{best_theta1:.8f}"
)

print(
    f"Best entangle = "
    f"{best_entangle:.8f}"
)

print(
    f"Best phi      = "
    f"{best_phi:.8f}"
)

print()
print(
    f"Optimizer energy = "
    f"{best_ideal_energy:.8f}"
)

print(
    f"Exact ground-state energy = "
    f"{EXACT_GROUND_ENERGY:.8f}"
)


# ============================================================
# Optimization error
# ============================================================

optimization_error = (
    abs(
        best_ideal_energy
        - EXACT_GROUND_ENERGY
    )
)


print()
print(
    f"Optimization / ansatz error = "
    f"{optimization_error:.8f}"
)


# ============================================================
# Final finite-shot verification
# ============================================================

(
    final_energy,
    final_z0,
    final_z1,
    final_zz,
    final_xx,
    final_z_counts,
    final_xx_counts
) = measured_energy(
    best_parameters,
    shots=SHOTS
)


# ============================================================
# Final results
# ============================================================

print()
print("Final Verification:")
print("===================")

print(
    f"<Z0>   = {final_z0:.8f}"
)

print(
    f"<Z1>   = {final_z1:.8f}"
)

print(
    f"<Z0Z1> = {final_zz:.8f}"
)

print(
    f"<X0X1> = {final_xx:.8f}"
)

print()
print(
    f"Ideal optimized energy = "
    f"{best_ideal_energy:.8f}"
)

print(
    f"Measured energy = "
    f"{final_energy:.8f}"
)

print(
    f"Exact ground energy = "
    f"{EXACT_GROUND_ENERGY:.8f}"
)


# ============================================================
# Measurement error
# ============================================================

measurement_error = abs(
    final_energy
    - best_ideal_energy
)


total_error = abs(
    final_energy
    - EXACT_GROUND_ENERGY
)


print()
print(
    f"Measurement error = "
    f"{measurement_error:.8f}"
)

print(
    f"Total error = "
    f"{total_error:.8f}"
)


# ============================================================
# Counts
# ============================================================

print()
print("Z-basis Counts:")
print(final_z_counts)

print()
print("XX-basis Counts:")
print(final_xx_counts)


# ============================================================
# Final summary
# ============================================================

print()
print("=================================")
print("VQE SUMMARY")
print("=================================")

print(
    f"Exact ground energy : "
    f"{EXACT_GROUND_ENERGY:.8f}"
)

print(
    f"Optimized ideal     : "
    f"{best_ideal_energy:.8f}"
)

print(
    f"Measured energy     : "
    f"{final_energy:.8f}"
)

print(
    f"Ansatz error        : "
    f"{optimization_error:.8f}"
)

print(
    f"Measurement error   : "
    f"{measurement_error:.8f}"
)

print(
    f"Total error         : "
    f"{total_error:.8f}"
)

print()

if optimization_error < 0.01:

    print(
        "SUCCESS: The entangling ansatz "
        "can reach the ground-state energy."
    )

else:

    print(
        "The ansatz still needs improvement."
    )