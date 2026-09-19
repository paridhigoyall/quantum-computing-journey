from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate
import numpy as np



# ============================================================
# Quantum Counting parameters
# ============================================================

N = 8
M = 2

print("N =", N)
print("M =", M)


# ============================================================
# Theoretical values
# ============================================================

theta = np.arcsin(np.sqrt(M / N))

print("\nTheoretical values:")
print("theta =", theta)
print("2 theta =", 2 * theta)

expected_phase = (2 * theta) / (2 * np.pi)

print("Expected normalized phase =", expected_phase)
print("Expected conjugate phase =", 1 - expected_phase)


# ============================================================
# Marked states
#
# We mark:
# |001>  -> decimal 1
# |010>  -> decimal 2
# ============================================================

marked_states = [1, 2]


# ============================================================
# Phase Oracle
#
# O|x> = -|x> for marked states
# O|x> =  |x> otherwise
# ============================================================

oracle = np.eye(N, dtype=complex)

for state in marked_states:
    oracle[state, state] = -1


print("\nOracle matrix:")
print(oracle)


# ============================================================
# Uniform state |s>
#
# |s> = 1/sqrt(N) * sum |x>
# ============================================================

s = np.ones(N, dtype=complex) / np.sqrt(N)


# ============================================================
# Diffusion operator
#
# D = 2|s><s| - I
# ============================================================

identity = np.eye(N, dtype=complex)

diffusion = (
    2 * np.outer(s, s.conj())
    - identity
)


# ============================================================
# Grover operator
#
# Q = D O
# ============================================================

Q = diffusion @ oracle


print("\nGrover operator:")
print(np.round(Q, 3))


# ============================================================
# Check unitarity
# ============================================================

print("\nIs Q unitary?")

print(
    np.allclose(
        Q.conj().T @ Q,
        identity
    )
)


# ============================================================
# Eigenvalue analysis
# ============================================================

eigenvalues = np.linalg.eigvals(Q)

print("\nGrover Eigenvalues:")

for value in eigenvalues:

    phase = np.angle(value)

    if phase < 0:
        phase += 2 * np.pi

    normalized_phase = phase / (2 * np.pi)

    print(
        f"Eigenvalue = {value:.4f} | "
        f"Phase = {phase:.4f} | "
        f"Normalized = {normalized_phase:.4f}"
    )


# ============================================================
# Identify the expected Grover phases
# ============================================================

print("\nExpected relevant phases:")

print(
    "Positive phase:",
    expected_phase
)

print(
    "Negative/conjugate phase:",
    1 - expected_phase
)
# ============================================================
# Quantum Counting + QPE
# ============================================================

counting_qubits = 4

qc = QuantumCircuit(
    counting_qubits + 3,
    counting_qubits
)

# ------------------------------------------------------------
# Prepare the search register in |s>
# ------------------------------------------------------------

for q in range(3):
    qc.h(counting_qubits + q)


# ------------------------------------------------------------
# Prepare counting register
# ------------------------------------------------------------

for q in range(counting_qubits):
    qc.h(q)


# ------------------------------------------------------------
# Controlled Q^(1), Q^(2), Q^(4), Q^(8)
# ------------------------------------------------------------

for j in range(counting_qubits):

    power = 2 ** j

    Q_power = np.linalg.matrix_power(
        Q,
        power
    )

    Q_gate = UnitaryGate(
        Q_power,
        label=f"Q^{power}"
    )

    controlled_Q = Q_gate.control(1)

    qc.append(
        controlled_Q,
        [
            j,
            counting_qubits,
            counting_qubits + 1,
            counting_qubits + 2
        ]
    )


# ============================================================
# Inverse QFT
# ============================================================

def inverse_qft(qc, qubits):

    n = len(qubits)

    # Reverse order
    for i in range(n // 2):
        qc.swap(
            qubits[i],
            qubits[n - i - 1]
        )

    # Controlled phase rotations
    for j in range(n):

        for m in range(j):

            angle = -np.pi / (
                2 ** (j - m)
            )

            qc.cp(
                angle,
                qubits[m],
                qubits[j]
            )

        qc.h(qubits[j])


inverse_qft(
    qc,
    list(range(counting_qubits))
)


# ============================================================
# Measurement
# ============================================================

qc.measure(
    range(counting_qubits),
    range(counting_qubits)
)


print("\nQuantum Counting / QPE Circuit:")
print(qc)


# ============================================================
# Run simulation
# ============================================================

simulator = AerSimulator()

compiled = transpile(
    qc,
    simulator
)

result = simulator.run(
    compiled,
    shots=2048
).result()

counts = result.get_counts()


print("\nMeasurement Results:")

for state, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{state}: {count}")


# ============================================================
# Decode dominant measurement
# ============================================================

dominant = max(
    counts,
    key=counts.get
)

value = int(dominant, 2)

raw_phase = value / (2 ** counting_qubits)

# Choose the phase between 0 and 1/2
phase = min(
    raw_phase,
    1 - raw_phase
)

theta_estimated = np.pi * phase

M_estimated = (
    N * np.sin(theta_estimated) ** 2
)


print("\nQuantum Counting Result:")
print("Dominant measurement:", dominant)
print("Measured integer:", value)
print("Raw phase:", raw_phase)
print("Selected phase:", phase)
print("Estimated theta:", theta_estimated)
print("Estimated M:", M_estimated)
print("True M:", M)

# ============================================================
# Decode dominant phase
# ============================================================

dominant = max(
    counts,
    key=counts.get
)

value = int(dominant, 2)

raw_phase = value / (2 ** counting_qubits)

# Handle the conjugate phase branch
phase = min(
    raw_phase,
    1 - raw_phase
)

theta_estimated = np.pi * phase

M_estimated = (
    N * np.sin(theta_estimated) ** 2
)

print("\nQuantum Counting Result:")
print("Dominant measurement:", dominant)
print("Measured integer:", value)
print("Raw phase:", raw_phase)
print("Selected phase:", phase)
print("Estimated theta:", theta_estimated)
print("Estimated M:", M_estimated)
print("True M:", M)

# ============================================================
# Simulation
# ============================================================

simulator = AerSimulator()

compiled = transpile(
    qc,
    simulator
)

result = simulator.run(
    compiled,
    shots=2048
).result()

counts = result.get_counts()


print("\nMeasurement Results:")

for state, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{state}: {count}")