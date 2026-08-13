import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate


# ============================================================
# Parameters
# ============================================================

N = 15
a = 2

counting_qubits = 4
target_qubits = 4

qc = QuantumCircuit(
    counting_qubits + target_qubits,
    counting_qubits
)


# ============================================================
# Step 1: Create modular multiplication unitary
#
# |y> -> |a*y mod N>
#
# For N = 15 and a = 2
# ============================================================

dimension = 2 ** target_qubits

U = np.zeros((dimension, dimension), dtype=complex)

for y in range(dimension):

    if y < N:
        new_y = (a * y) % N
    else:
        # Keep |15> unchanged so that the operation
        # remains a valid permutation on all 16 states.
        new_y = y

    U[new_y, y] = 1


mod_mult = UnitaryGate(U, label="×2 mod 15")


# ============================================================
# Step 2: Prepare target register in |1>
# ============================================================

qc.x(counting_qubits)


# ============================================================
# Step 3: Put counting register into superposition
# ============================================================

for qubit in range(counting_qubits):
    qc.h(qubit)


# ============================================================
# Step 4: Controlled modular exponentiation
#
# U^(2^j)
#
# This creates:
#
# |x>|1> -> |x>|2^x mod 15>
# ============================================================

for j in range(counting_qubits):

    power = 2 ** j

    U_power = np.linalg.matrix_power(U, power)

    controlled_U = UnitaryGate(
        U_power,
        label=f"×2^{power} mod 15"
    ).control(1)

    qc.append(
        controlled_U,
        [j] + list(range(counting_qubits, counting_qubits + target_qubits))
    )


# ============================================================
# Step 5: Inverse QFT on counting register
# ============================================================

def inverse_qft(qc, qubits):

    n = len(qubits)

    # Reverse order
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - i - 1])

    # Inverse QFT
    for j in range(n):
        qc.h(qubits[j])

        for m in range(j + 1, n):
            angle = -np.pi / (2 ** (m - j))

            qc.cp(
                angle,
                qubits[m],
                qubits[j]
            )


inverse_qft(qc, list(range(counting_qubits)))


# ============================================================
# Step 6: Measurement
# ============================================================

qc.measure(
    range(counting_qubits),
    range(counting_qubits)
)


# ============================================================
# Display circuit
# ============================================================

print("Shor Period-Finding Circuit:")
print(qc)


# ============================================================
# Step 7: Simulation
# ============================================================

simulator = AerSimulator()

compiled = transpile(
    qc,
    simulator
)

result = simulator.run(
    compiled,
    shots=1024
).result()

counts = result.get_counts()


print("\nMeasurement Results:")

for state, count in sorted(counts.items()):
    print(f"{state}: {count}")