import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate


# ============================================================
# Parameters
# ============================================================

a = 0.25

theta = np.arcsin(np.sqrt(a))

counting_qubits = 8

print("Amplitude a:", a)
print("Theta:", theta)
print(
    "Expected eigenphase fraction:",
    2 * theta / (2 * np.pi)
)


# ============================================================
# State preparation A = Ry(2*theta)
# ============================================================

A = QuantumCircuit(1)

A.ry(2 * theta, 0)


# Matrix of Ry(2*theta)
A_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
], dtype=complex)


# ============================================================
# Define Z reflections
# ============================================================

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)


# ============================================================
# Construct Grover operator
#
# Q = A Z A† Z
# ============================================================

A_dagger = A_matrix.conj().T

Q_matrix = (
    A_matrix
    @ Z
    @ A_dagger
    @ Z
)


Q_gate = UnitaryGate(
    Q_matrix,
    label="Q"
)


# ============================================================
# Display Grover matrix
# ============================================================

print("\nGrover operator matrix:")
print(Q_matrix)


# ============================================================
# Create QAE / QPE circuit
# ============================================================

qc = QuantumCircuit(
    counting_qubits + 1,
    counting_qubits
)


# ============================================================
# Prepare amplitude state
# ============================================================

qc.ry(
    2 * theta,
    counting_qubits
)


# ============================================================
# Counting register superposition
# ============================================================

for q in range(counting_qubits):
    qc.h(q)


# ============================================================
# Controlled powers of Q
# ============================================================

for j in range(counting_qubits):

    power = 2 ** j

    Q_power = np.linalg.matrix_power(
        Q_matrix,
        power
    )

    controlled_Q = UnitaryGate(
        Q_power,
        label=f"Q^{power}"
    ).control(1)

    qc.append(
        controlled_Q,
        [j, counting_qubits]
    )


# ============================================================
# Inverse QFT
# ============================================================

def inverse_qft(qc, qubits):

    n = len(qubits)

    # Reverse qubit order
    for i in range(n // 2):
        qc.swap(
            qubits[i],
            qubits[n - i - 1]
        )

    # Inverse QFT
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


# ============================================================
# Display circuit
# ============================================================

print("\nQAE / QPE Circuit:")
print(qc)


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

# ============================================================
# Decode dominant phase
# ============================================================

# ============================================================
# Decode phase and handle both QAE phase branches
# ============================================================

most_likely = max(
    counts,
    key=counts.get
)

measured_value = int(
    most_likely,
    2
)

raw_phase = measured_value / (2 ** counting_qubits)

# QAE can produce phi or 1 - phi.
# We use the phase in [0, 1/2].
phase = min(
    raw_phase,
    1 - raw_phase
)

theta_estimated = np.pi * phase

a_estimated = np.sin(theta_estimated) ** 2


print("\nPhase Estimation:")
print("Measured bitstring:", most_likely)
print("Measured integer:", measured_value)
print("Raw phase:", raw_phase)
print("Selected phase:", phase)
print("Estimated theta:", theta_estimated)
print("Estimated amplitude:", a_estimated)

print("\nTrue amplitude:", a)

print(
    "Absolute error:",
    abs(a_estimated - a)
)