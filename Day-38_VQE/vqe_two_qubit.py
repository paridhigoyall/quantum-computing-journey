import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# ============================================================
# Day 39 - Generic Two-Qubit VQE
# H = 0.5 Z0 + 0.5 Z1 + 0.8 Z0Z1 + 0.2 X0X1
# ============================================================

HAMILTONIAN = [
    ("Z0", 0.5),
    ("Z1", 0.5),
    ("Z0Z1", 0.8),
    ("X0X1", 0.2),
]

EXACT_GROUND_STATE_ENERGY = -1.0
OPTIMIZATION_SHOTS = 4096
FINAL_SHOTS = 16384

simulator = AerSimulator()


# ============================================================
# Variational ansatz
# ============================================================

def build_ansatz(theta0, theta1, theta2, theta3):
    qc = QuantumCircuit(2)
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.cx(0, 1)
    qc.ry(theta2, 0)
    qc.ry(theta3, 1)
    return qc


# ============================================================
# Generic Pauli measurement
# Supported: Z0, Z1, Z0Z1, X0, X1, X0X1
# ============================================================

def measure_pauli(theta0, theta1, theta2, theta3, pauli_term,
                  shots=OPTIMIZATION_SHOTS):

    qc = build_ansatz(theta0, theta1, theta2, theta3)

    # X measurement is converted to Z measurement with H.
    if "X0" in pauli_term:
        qc.h(0)
    if "X1" in pauli_term:
        qc.h(1)

    qc.measure_all()

    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    total = 0

    for bitstring, count in counts.items():
        # Qiskit displays two qubits as q1 q0.
        q1 = int(bitstring[0])
        q0 = int(bitstring[1])

        eigenvalue = 1

        if "Z0" in pauli_term or "X0" in pauli_term:
            eigenvalue *= 1 if q0 == 0 else -1

        if "Z1" in pauli_term or "X1" in pauli_term:
            eigenvalue *= 1 if q1 == 0 else -1

        total += eigenvalue * count

    return total / shots, counts


# ============================================================
# Generic Hamiltonian evaluation
# ============================================================

def evaluate_hamiltonian(theta0, theta1, theta2, theta3,
                         shots=OPTIMIZATION_SHOTS):

    energy = 0.0
    measurements = {}

    for pauli_term, coefficient in HAMILTONIAN:
        expectation, counts = measure_pauli(
            theta0, theta1, theta2, theta3,
            pauli_term, shots
        )

        contribution = coefficient * expectation
        energy += contribution

        measurements[pauli_term] = {
            "coefficient": coefficient,
            "expectation": expectation,
            "contribution": contribution,
            "counts": counts,
        }

    return energy, measurements


# ============================================================
# COBYLA objective
# ============================================================

def objective(params):
    energy, measurements = evaluate_hamiltonian(*params)

    z0 = measurements["Z0"]["expectation"]
    z1 = measurements["Z1"]["expectation"]
    zz = measurements["Z0Z1"]["expectation"]
    xx = measurements["X0X1"]["expectation"]

    print(
        f"theta0={params[0]:.4f}, theta1={params[1]:.4f}, "
        f"theta2={params[2]:.4f}, theta3={params[3]:.4f} | "
        f"<Z0>={z0:+.4f}, <Z1>={z1:+.4f}, "
        f"<ZZ>={zz:+.4f}, <XX>={xx:+.4f} | E={energy:+.4f}"
    )

    return energy


def print_hamiltonian():
    pieces = []
    for term, coefficient in HAMILTONIAN:
        pieces.append(f"{coefficient:+.3f}{term}")
    expression = " ".join(pieces)
    if expression.startswith("+"):
        expression = expression[1:].lstrip()
    print(f"H = {expression}")


# ============================================================
# Main program
# ============================================================

initial_parameters = np.array([0.5, 0.5, 0.0, 0.0])

print("Day 39 - Generic Two-Qubit VQE")
print("==============================")
print_hamiltonian()
print(f"Exact ground-state energy = {EXACT_GROUND_STATE_ENERGY:.6f}")

print("\nInitial Parameters:")
for i, value in enumerate(initial_parameters):
    print(f"theta{i} = {value:.6f}")

initial_energy, initial_measurements = evaluate_hamiltonian(
    *initial_parameters, shots=FINAL_SHOTS
)

print("\nInitial Measurements:")
for term, data in initial_measurements.items():
    print(
        f"{term:<5} | coefficient={data['coefficient']:+.3f} | "
        f"<P>={data['expectation']:+.6f} | "
        f"contribution={data['contribution']:+.6f}"
    )
print(f"Initial Energy = {initial_energy:.6f}")


print("\nCOBYLA Optimization:")
result = minimize(
    objective,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": 60, "rhobeg": 0.5},
)

best = result.x

print("\nOptimization Complete:")
for i, value in enumerate(best):
    print(f"Best theta{i} = {value:.6f}")
print(f"Optimizer energy = {result.fun:.6f}")
print(f"Exact ground-state energy = {EXACT_GROUND_STATE_ENERGY:.6f}")


final_energy, final_measurements = evaluate_hamiltonian(
    *best, shots=FINAL_SHOTS
)

print("\nFinal Verification:")
for term, data in final_measurements.items():
    print(
        f"{term:<5} | <P>={data['expectation']:+.6f} | "
        f"contribution={data['contribution']:+.6f}"
    )

print(f"\nFinal Energy = {final_energy:.6f}")
print(f"Exact Ground-State Energy = {EXACT_GROUND_STATE_ENERGY:.6f}")
print(
    f"Absolute Energy Error = "
    f"{abs(final_energy - EXACT_GROUND_STATE_ENERGY):.6f}"
)

print("\nVQE Summary:")
if final_energy <= EXACT_GROUND_STATE_ENERGY + 0.02:
    print("The VQE result is very close to the exact ground-state energy.")
else:
    print("The VQE result is not yet very close to the exact ground-state energy.")

print(
    "The Hamiltonian is now represented as Pauli terms, so the "
    "measurement and energy logic can be reused for new Hamiltonians."
)